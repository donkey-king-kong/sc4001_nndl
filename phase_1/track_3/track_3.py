# -*- coding: utf-8 -*-
r"""track_3.ipynb

Same code as track_3.ipynb, cell by cell (markdown cells as docstrings).

# Track 3: Object Swaps and Ranking Loss

Builds the CHAIR-compatible object co-occurrence matrix, generates structured and random-swap negatives, spot-checks 100 negatives,
then trains ClipCap with an extra ranking term and compares it to the Track 2 baseline.

Loss for a batch of captions, each with one object-swap negative (same image prefix):

$$\mathcal{L} = \text{NLL}(\text{positives}) + \lambda \cdot \frac{1}{|N|}\sum_{i \in N} \max(0,\ m + \text{nll}_i(\text{pos}) - \text{nll}_i(\text{neg}))$$

- NLL(positives) is exactly Track 2's `caption_loss` (token mean, **no EOS**, as in ClipCap and the baseline). $\text{nll}_i$ is the per-token mean NLL of caption $i$; $N$ = captions that have a valid negative.
- Positives and negatives go through GPT-2 in **one batched forward pass** that shares one mapping-network pass.
- Training is Track 2's `train_run` unchanged (same checkpointing, resume, seeds, batch order, fp16, val loss); decoding is `decode_split` and scoring is Track 1's `evaluate`.
  With $\lambda = 0$ the negatives are dropped and every step is the baseline step.
- Expect each run to take about 1.5-2x the baseline (the negative doubles the GPT-2 batch).
- Comparison: Track 1's `paired_bootstrap` on the `.per_image.json` files, baseline seed $s$ vs Track 3 seed $s$.

**How to run:** GPU runtime, then Runtime > Run all. After a disconnect, Run all again: finished runs are skipped and training resumes from its checkpoint.

# Setup

## Installs
"""

"""## Imports
"""

import os
import sys
import json
import pickle
import random
import shutil
import subprocess
import time
import zipfile
import warnings
import logging

import nltk
import inflect
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import GPT2TokenizerFast, get_linear_schedule_with_warmup

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

# ClipCap's train.py imports AdamW from transformers, removed in newer versions.
# Setting transformers.AdamW no longer works on recent versions, so patch the import line once.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.environ.get(
    'SC4001_PROJECT_ROOT',
    os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..')),
)
CLIP_ROOT = os.environ.get(
    'CLIP_PREFIX_CAPTION_ROOT',
    os.path.join(PROJECT_ROOT, 'CLIP_prefix_caption'),
)
TRAIN_PY = os.path.join(CLIP_ROOT, 'train.py')
if not os.path.exists(TRAIN_PY):
    raise FileNotFoundError(
        f'ClipCap checkout not found at {CLIP_ROOT}. '
        'Set CLIP_PREFIX_CAPTION_ROOT to the checkout directory.'
    )
OLD_IMPORT = 'from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup'
NEW_IMPORT = ('from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup\n'
              'from torch.optim import AdamW')
with open(TRAIN_PY) as fh:
    train_src = fh.read()
if OLD_IMPORT in train_src:
    with open(TRAIN_PY, 'w') as fh:
        fh.write(train_src.replace(OLD_IMPORT, NEW_IMPORT))
    print('patched ClipCap train.py AdamW import')
sys.path.append(CLIP_ROOT)
from train import ClipCaptionPrefix, MappingType

warnings.filterwarnings('ignore')
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

# stop Java warnings from corrupting pycocoevalcap output (tokenizer, METEOR)
os.environ['JAVA_TOOL_OPTIONS'] = '-Xlog:disable'

for pkg in ['punkt', 'punkt_tab', 'wordnet', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    nltk.download(pkg, quiet=True)

if torch.cuda.is_available():
    device = 'cuda'
    print('using:', torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    device = 'mps'
    print('using: Apple Silicon GPU')
else:
    device = 'cpu'
    print('using: CPU')

print('project folder:', PROJECT_ROOT)
TRACK1_ROOT = os.path.join(PROJECT_ROOT, 'phase_1', 'track_1')
TRACK2_ROOT = os.path.join(PROJECT_ROOT, 'phase_1', 'track_2')
TRACK3_ROOT = os.path.join(PROJECT_ROOT, 'phase_1', 'track_3')

# Track 1 inputs (read only)
DATA_DIR = os.path.join(TRACK1_ROOT, 'data')
FEAT_DIR = os.path.join(TRACK1_ROOT, 'features')
EVAL_DIR = os.path.join(TRACK1_ROOT, 'eval')
PRETRAINED_CKPT = os.path.join(TRACK1_ROOT, 'models', 'clipcap_transformer_coco_pretrained.pt')

# Track 2 baseline results (read only): clipcap_baseline_seed{s}_val.per_image.json
BASELINE_RESULTS = os.path.join(TRACK2_ROOT, 'results')

# Track 3 outputs; evaluate() writes its .eval.json and .per_image.json files to RESULTS_DIR
RESULTS_DIR = os.path.join(TRACK3_ROOT, 'results')
MODELS_DIR = os.path.join(TRACK3_ROOT, 'models')
RUNS_DIR = os.path.join(TRACK3_ROOT, 'runs')
CACHE_DIR = os.path.join(TRACK3_ROOT, 'cache')
for d in [RESULTS_DIR, MODELS_DIR, RUNS_DIR, CACHE_DIR, os.path.join(TRACK3_ROOT, 'eval')]:
    os.makedirs(d, exist_ok=True)

for p in [os.path.join(FEAT_DIR, 'coco_rn50x4_train.pkl'), os.path.join(FEAT_DIR, 'coco_rn50x4_val.pkl'),
          PRETRAINED_CKPT, os.path.join(EVAL_DIR, 'chair_gt_train.pkl'), os.path.join(EVAL_DIR, 'chair_gt_val.pkl'),
          os.path.join(EVAL_DIR, 'synonyms.txt'), os.path.join(EVAL_DIR, 'coco_objects.json'),
          os.path.join(DATA_DIR, 'karpathy_index.json')]:
    assert os.path.exists(p), 'missing: ' + p
print('inputs: ok')

"""# Helpers

Track 2 helpers: same code as `phase_1/track_2/track_2.py` (copied from `track_2_worker0.ipynb`). Do not edit here.

## Constants
"""

MAX_TOKENS = 30  # team rule: greedy, max 30 tokens, stop at '.'

"""## General helpers
"""

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(obj, path):
    # tmp then rename, so a disconnect never leaves a half file
    with open(path + '.tmp', 'w') as fh:
        json.dump(obj, fh)
    os.replace(path + '.tmp', path)


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def save_torch(obj, path):
    torch.save(obj, path + '.tmp')
    os.replace(path + '.tmp', path)

"""## Data
"""

def load_features(pkl_path):
    # ClipCap pkl: {'clip_embedding': [n, 640] float16, 'captions': [{image_id, caption, clip_embedding}]}
    with open(pkl_path, 'rb') as fh:
        data = pickle.load(fh)

    features = data['clip_embedding']
    captions = data['captions']
    row_image = [None] * features.shape[0]

    for c in captions:
        row_image[c['clip_embedding']] = c['image_id']

    assert all(i is not None for i in row_image), 'feature row with no caption'
    return {'features': features, 'captions': captions, 'row_image': row_image}


def subset_images(split_data, n_images, seed=0):
    # random subset of images (with all their captions), rows renumbered
    n = split_data['features'].shape[0]
    rows = sorted(random.Random(seed).sample(range(n), n_images))
    new_row = {}

    for j in range(len(rows)):
        new_row[rows[j]] = j

    captions = []
    for c in split_data['captions']:
        if c['clip_embedding'] in new_row:
            captions.append({'image_id': c['image_id'], 'caption': c['caption'],
                             'clip_embedding': new_row[c['clip_embedding']]})

    row_image = [split_data['row_image'][r] for r in rows]
    return {'features': split_data['features'][rows], 'captions': captions, 'row_image': row_image}


def tokenize_captions(captions, tokenizer, cache_path=None):
    # list of GPT-2 token id lists, one per caption (no EOS, same as ClipCap)
    texts = [c['caption'] for c in captions]

    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, 'rb') as fh:
            cached = pickle.load(fh)
        if cached['texts'] == texts:
            return cached['tokens']
        print('token cache does not match captions, rebuilding')

    tokens = tokenizer(texts, add_special_tokens=False)['input_ids']
    assert all(len(t) > 0 for t in tokens), 'empty caption'

    if cache_path is not None:
        with open(cache_path + '.tmp', 'wb') as fh:
            pickle.dump({'texts': texts, 'tokens': tokens}, fh)
        os.replace(cache_path + '.tmp', cache_path)
    return tokens


def max_seq_len(token_lists):
    # ClipCap's rule: mean + 10 std, capped at the longest caption
    lengths = torch.tensor([len(t) for t in token_lists]).float()
    return min(int(lengths.mean() + lengths.std() * 10), int(lengths.max()))


class CaptionDataset(Dataset):
    # one item per caption: (token ids, feature row)
    def __init__(self, split_data, token_lists, max_len):
        assert len(token_lists) == len(split_data['captions'])
        self.rows = [c['clip_embedding'] for c in split_data['captions']]
        self.tokens = [t[:max_len] for t in token_lists]
        self.features = split_data['features']
        self.max_len = max_len

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        return self.tokens[i], self.rows[i]


def collate(batch):
    # pad to the longest caption in the batch; padded labels are -100 (ignored)
    longest = max(len(t) for t, _ in batch)
    tokens = torch.zeros(len(batch), longest, dtype=torch.long)
    labels = torch.full((len(batch), longest), -100, dtype=torch.long)

    for i, (t, _) in enumerate(batch):
        tokens[i, :len(t)] = torch.tensor(t)
        labels[i, :len(t)] = torch.tensor(t)

    rows = torch.tensor([r for _, r in batch], dtype=torch.long)
    return tokens, labels, rows

"""## Model
"""

def build_model(mapping, prefix_length, clip_length=None, num_layers=8, prefix_size=640):
    # ClipCap model with GPT-2 frozen; needs CLIP_prefix_caption on sys.path
    from train import ClipCaptionPrefix, MappingType

    if mapping == 'mlp':
        mapping_type = MappingType.MLP
    elif mapping == 'transformer':
        mapping_type = MappingType.Transformer
    else:
        raise ValueError(mapping)

    if clip_length is None:
        clip_length = prefix_length

    model = ClipCaptionPrefix(prefix_length, clip_length=clip_length, prefix_size=prefix_size,
                              num_layers=num_layers, mapping_type=mapping_type)
    # no grads for GPT-2 weights; grads still flow through it to the prefix
    model.gpt.requires_grad_(False)
    return model


def n_mapping_params(model):
    return sum(p.numel() for p in model.clip_project.parameters())


def load_pretrained_checkpoint(model, path):
    # official checkpoint: 0 missing keys, only stale GPT-2 attention buffers unexpected
    state = torch.load(path, map_location='cpu')
    result = model.load_state_dict(state, strict=False)
    assert len(result.missing_keys) == 0, result.missing_keys
    for key in result.unexpected_keys:
        assert key.endswith('.attn.bias') or key.endswith('.attn.masked_bias'), key
    return model


def normalise_prefix(features):
    # raw CLIP features -> float32, unit length per row
    prefix = features.float()
    return prefix / prefix.norm(dim=-1, keepdim=True)


def prefix_embeddings(model, features):
    prefix = normalise_prefix(features)
    return model.clip_project(prefix).reshape(prefix.shape[0], model.prefix_length, -1)


def caption_loss(model, tokens, labels, features, reduction='mean'):
    # teacher-forced NLL of the caption tokens, same targets as ClipCap's train.py:
    # position prefix_length-1 predicts token 0, and so on
    prefix_embeds = prefix_embeddings(model, features)
    text_embeds = model.gpt.transformer.wte(tokens)
    inputs = torch.cat((prefix_embeds.to(text_embeds.dtype), text_embeds), dim=1)

    # run lm_head only on the positions that predict caption tokens
    hidden = model.gpt.transformer(inputs_embeds=inputs).last_hidden_state
    p = model.prefix_length
    logits = model.gpt.lm_head(hidden[:, p - 1:-1])

    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           ignore_index=-100, reduction=reduction)

"""## Greedy decoding (same algorithm as Track 1's greedy_decode_batch)
"""

def stop_token_id(tokenizer):
    return tokenizer.encode('.')[0]


@torch.no_grad()
def greedy_decode_batch(model, tokenizer, features, device, max_tokens=MAX_TOKENS):
    # features: [B, 640] raw CLIP vectors
    stop_id = stop_token_id(tokenizer)
    n = features.shape[0]
    embeds = prefix_embeddings(model, features.to(device))

    token_ids = [[] for _ in range(n)]
    finished = torch.zeros(n, dtype=torch.bool, device=device)

    for _ in range(max_tokens):
        logits = model.gpt(inputs_embeds=embeds).logits[:, -1, :]
        next_ids = logits.argmax(dim=-1)
        next_list = next_ids.tolist()

        for b in range(n):
            # keep tokens up to and including the first stop
            if not finished[b]:
                token_ids[b].append(next_list[b])

        finished = finished | (next_ids == stop_id)
        if bool(finished.all()):
            break

        next_embed = model.gpt.transformer.wte(next_ids).unsqueeze(1)
        embeds = torch.cat((embeds, next_embed), dim=1)

    return [tokenizer.decode(t).strip() for t in token_ids]


def decode_split(model, tokenizer, split_data, out_path, device, batch_size=64):
    # captions for every image in the split -> [{image_id, caption}], resumable
    model.eval()
    partial_path = out_path.replace('.json', '.partial.json')
    row_image = split_data['row_image']
    features = split_data['features']

    if os.path.exists(out_path):
        return read_json(out_path)

    done = {}
    if os.path.exists(partial_path):
        for p in read_json(partial_path):
            done[p['image_id']] = p['caption']

    n_rows = features.shape[0]
    starts = list(range(0, n_rows, batch_size))

    for k, start in enumerate(tqdm(starts, desc='captioning')):
        rows = [j for j in range(start, min(start + batch_size, n_rows)) if row_image[j] not in done]
        if len(rows) == 0:
            continue

        caps = greedy_decode_batch(model, tokenizer, features[rows], device)
        for i in range(len(rows)):
            done[row_image[rows[i]]] = caps[i]

        if k % 10 == 9:
            write_json([{'image_id': i, 'caption': c} for i, c in done.items()], partial_path)

    out = [{'image_id': int(row_image[j]), 'caption': done[row_image[j]]} for j in range(n_rows)]
    assert len(out) == n_rows
    write_json(out, out_path)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    return out

"""## Training
"""

def default_config(**overrides):
    # ClipCap train.py defaults: 10 epochs, bs 40, lr 2e-5, 5000 warmup steps
    cfg = {
        'mapping': 'transformer',
        'prefix_length': 40,
        'clip_length': None,      # None -> same as prefix_length
        'num_layers': 8,
        'epochs': 10,
        'train_steps': None,      # if set, train exactly this many steps instead of `epochs`
        'batch_size': 40,
        'lr': 2e-5,
        'warmup_steps': 5000,
        'seed': 0,
        'amp': True,              # bf16 if the GPU has it, else fp16 with loss scaling
        'ckpt_minutes': 10,       # save a resume checkpoint this often (also at every epoch end)
        'log_every': 500,
    }
    for k in overrides:
        assert k in cfg, f'unknown config key {k}'
        cfg[k] = overrides[k]
    return cfg


def config_name(cfg):
    return f"{cfg['mapping']}_p{cfg['prefix_length']}"


def model_from_config(cfg):
    return build_model(cfg['mapping'], cfg['prefix_length'], cfg['clip_length'], cfg['num_layers'])


# keys that must match when resuming a run
RUN_KEYS = ['mapping', 'prefix_length', 'clip_length', 'num_layers', 'epochs', 'train_steps',
            'batch_size', 'lr', 'warmup_steps', 'seed']


def total_train_steps(cfg, n_train):
    if cfg['train_steps'] is not None:
        return cfg['train_steps']
    return cfg['epochs'] * (n_train // cfg['batch_size'])


def amp_dtype(device):
    # bf16 only on GPUs with native support (A100, L4); T4 uses fp16 with loss scaling
    if device == 'cuda' and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    return torch.float16


def epoch_batches(n_items, batch_size, seed, epoch):
    # same shuffle for a given (seed, epoch), so a resumed run sees the same batches
    g = torch.Generator().manual_seed(seed * 100003 + epoch)
    perm = torch.randperm(n_items, generator=g).tolist()
    n_steps = n_items // batch_size  # drop_last, as in ClipCap
    return [perm[i * batch_size:(i + 1) * batch_size] for i in range(n_steps)]


@torch.no_grad()
def dataset_loss(model, dataset, features_gpu, device, cfg, batch_size=200):
    # mean NLL per caption token over a whole dataset
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    total = 0.0
    count = 0

    for tokens, labels, rows in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        with torch.autocast('cuda', dtype=amp_dtype(device), enabled=cfg['amp'] and device == 'cuda'):
            loss_sum = caption_loss(model, tokens, labels, features_gpu[rows.to(device)], reduction='sum')
        total += float(loss_sum)
        count += int((labels != -100).sum())

    return total / count


def train_run(cfg, train_set, val_set, run_dir, device, max_steps=None, deadline=None, log=print):
    """Train one config. Resumes from run_dir/last.pt; returns (model, run_log).

    max_steps stops early (smoke tests); such runs are not marked finished.
    deadline (a time.time() value) saves a checkpoint and stops, for job time limits;
    the run is not marked finished and resumes from that checkpoint next time.
    """
    os.makedirs(run_dir, exist_ok=True)
    state_path = os.path.join(run_dir, 'last.pt')
    final_path = os.path.join(run_dir, 'mapping_final.pt')
    log_path = os.path.join(run_dir, 'run_log.json')

    set_seed(cfg['seed'])
    model = model_from_config(cfg)

    # finished earlier: load and return
    if os.path.exists(final_path) and os.path.exists(log_path):
        saved = torch.load(final_path, map_location='cpu')
        model.clip_project.load_state_dict(saved['clip_project'])
        log('run already finished:', run_dir)
        return model.to(device), read_json(log_path)

    model = model.to(device)
    bs = cfg['batch_size']
    steps_per_epoch = len(train_set) // bs
    total_steps = total_train_steps(cfg, len(train_set))
    n_epochs = -(-total_steps // steps_per_epoch)  # ceil: the last epoch may be partial

    optimizer = torch.optim.AdamW(model.clip_project.parameters(), lr=cfg['lr'])
    scheduler = get_linear_schedule_with_warmup(optimizer, cfg['warmup_steps'], total_steps)
    use_amp = cfg['amp'] and device == 'cuda'
    dtype = amp_dtype(device)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and dtype == torch.float16)

    run_log = {
        'config': cfg,
        'run_dir': run_dir,
        'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
        'amp_dtype': str(dtype) if use_amp else 'float32',
        'mapping_params': n_mapping_params(model),
        'n_train_captions': len(train_set),
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'train_loss': [],    # [global_step, mean loss since last log]
        'epochs': [],        # one record per finished epoch
        'train_seconds': 0.0,
        'finished': False,
    }
    start_epoch = 0
    start_step = 0

    if os.path.exists(state_path):
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        for k in RUN_KEYS:
            assert state['run_log']['config'][k] == cfg[k], f'resume config mismatch on {k}'
        model.clip_project.load_state_dict(state['clip_project'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        start_epoch = state['epoch']
        start_step = state['step']
        run_log = state['run_log']
        # record every GPU type the run used ('gpu' keeps the first one)
        gpu_now = torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'
        run_log.setdefault('gpus', [run_log['gpu']])
        if run_log['gpus'][-1] != gpu_now:
            run_log['gpus'].append(gpu_now)
        log(f'resuming {run_dir} at epoch {start_epoch} step {start_step}')

    def save_state(epoch, step):
        save_torch({
            'clip_project': model.clip_project.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'step': step,
            'run_log': run_log,
        }, state_path)

    features_gpu = train_set.features.to(device)
    val_features_gpu = val_set.features.to(device) if val_set is not None else None
    global_step = start_epoch * steps_per_epoch + start_step
    stopped_early = False

    for epoch in range(start_epoch, n_epochs):
        skip = start_step if epoch == start_epoch else 0
        batches = epoch_batches(len(train_set), bs, cfg['seed'], epoch)[skip:]
        loader = DataLoader(train_set, batch_sampler=batches, collate_fn=collate, num_workers=0)

        model.train()
        t_epoch = time.time()
        t_saved = time.time()
        loss_sum = 0.0
        loss_n = 0
        epoch_loss_sum = 0.0
        epoch_loss_n = 0
        step = skip

        bar = tqdm(loader, initial=skip, total=steps_per_epoch, desc=f'{config_name(cfg)} s{cfg["seed"]} ep{epoch}')
        for tokens, labels, rows in bar:
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            prefix = features_gpu[rows.to(device)]

            with torch.autocast('cuda', dtype=dtype, enabled=use_amp):
                loss = caption_loss(model, tokens, labels, prefix)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            step += 1
            global_step += 1
            value = float(loss.detach())
            loss_sum += value
            loss_n += 1
            epoch_loss_sum += value
            epoch_loss_n += 1

            if global_step % cfg['log_every'] == 0:
                run_log['train_loss'].append([global_step, loss_sum / loss_n])
                bar.set_postfix(loss=f'{loss_sum / loss_n:.3f}')
                loss_sum = 0.0
                loss_n = 0

            if global_step >= total_steps:
                break  # planned end of training, possibly mid-epoch

            if max_steps is not None and global_step >= max_steps:
                stopped_early = True
                break

            if time.time() - t_saved > cfg['ckpt_minutes'] * 60:
                run_log['train_seconds'] += time.time() - t_saved
                save_state(epoch, step)
                t_saved = time.time()

            if deadline is not None and time.time() > deadline:
                run_log['train_seconds'] += time.time() - t_saved
                save_state(epoch, step)
                t_saved = time.time()
                log(f'time limit reached: saved checkpoint at epoch {epoch} step {step}')
                stopped_early = True
                break

        run_log['train_seconds'] += time.time() - t_saved
        if stopped_early:
            break

        epoch_seconds = time.time() - t_epoch
        record = {
            'epoch': epoch,
            'train_loss': epoch_loss_sum / max(epoch_loss_n, 1),
            'session_seconds': epoch_seconds,
            'session_steps': epoch_loss_n,
            'session_sec_per_step': epoch_seconds / max(epoch_loss_n, 1),
        }
        if val_set is not None:
            record['val_loss'] = dataset_loss(model, val_set, val_features_gpu, device, cfg)
        run_log['epochs'].append(record)
        log(f'epoch {epoch}: ' + ', '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
                                          for k, v in record.items()))
        save_state(epoch + 1, 0)
        model.train()

    if stopped_early:
        run_log['stopped_at_step'] = global_step
        return model, run_log

    run_log['finished'] = True
    save_torch({'config': cfg, 'clip_project': model.clip_project.state_dict()}, final_path)
    write_json(run_log, log_path)
    if os.path.exists(state_path):
        os.remove(state_path)  # big file, not needed once finished
    return model, run_log


def benchmark(cfg, train_set, device, n_steps=100, warmup=10):
    # time training steps for one config, nothing saved; returns sec/step and peak memory
    set_seed(cfg['seed'])
    model = model_from_config(cfg).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.clip_project.parameters(), lr=cfg['lr'])
    use_amp = cfg['amp'] and device == 'cuda'
    dtype = amp_dtype(device)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and dtype == torch.float16)
    features_gpu = train_set.features.to(device)
    batches = epoch_batches(len(train_set), cfg['batch_size'], cfg['seed'], 0)[:n_steps + warmup]
    loader = DataLoader(train_set, batch_sampler=batches, collate_fn=collate)

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    t0 = None
    for i, (tokens, labels, rows) in enumerate(loader):
        if i == warmup:
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast('cuda', dtype=dtype, enabled=use_amp):
            loss = caption_loss(model, tokens.to(device), labels.to(device), features_gpu[rows.to(device)])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    if device == 'cuda':
        torch.cuda.synchronize()
    sec_per_step = (time.time() - t0) / n_steps
    steps_per_epoch = len(train_set) // cfg['batch_size']
    result = {
        'config': config_name(cfg),
        'mapping_params_M': n_mapping_params(model) / 1e6,
        'sec_per_step': sec_per_step,
        'hours_per_epoch': sec_per_step * steps_per_epoch / 3600,
        'peak_mem_GB': torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else 0.0,
        # resume checkpoint = weights + 2 Adam moments, fp32
        'ckpt_GB': n_mapping_params(model) * 4 * 3 / 1e9,
    }
    del model, optimizer
    if device == 'cuda':
        torch.cuda.empty_cache()
    return result


def load_trained(run_dir, device):
    # rebuild a finished run's model from its mapping weights
    saved = torch.load(os.path.join(run_dir, 'mapping_final.pt'), map_location='cpu')
    model = model_from_config(saved['config'])
    model.clip_project.load_state_dict(saved['clip_project'])
    return model.to(device).eval()

"""## Track 1 evaluation helpers

Copied verbatim from `track_1.ipynb` ("Annotations > Helper Functions" `download`,
"CHAIR and Object List > Helper Functions" and "Evaluation > Helper Functions"), so scores match Track 1 exactly. Do not edit here.
"""

def download(url, out_dir):
    subprocess.run(['wget', '-c', '-q', '--show-progress', '-P', out_dir, url], check=True)
    return os.path.join(out_dir, os.path.basename(url))


def extract_members(zip_path, names, out_dir):
    # copy selected files out of a zip, tmp then rename so no half files
    with zipfile.ZipFile(zip_path) as z:
        lookup = {}
        for member in z.namelist():
            lookup[os.path.basename(member)] = member
        for name in names:
            dst = os.path.join(out_dir, name)
            with z.open(lookup[name]) as src, open(dst + '.tmp', 'wb') as out:
                shutil.copyfileobj(src, out)
            os.replace(dst + '.tmp', dst)


def norm_split(s):
    # restval counts as train
    if s in ('train', 'restval'):
        return 'train'
    else:
        return s

SYN_URL = 'https://raw.githubusercontent.com/LisaAnne/Hallucination/master/data/synonyms.txt'
SYN_PATH = os.path.join(EVAL_DIR, 'synonyms.txt')

# small text file downloaded once
if not os.path.exists(SYN_PATH):
    print('downloading synonyms.txt...')
    download(SYN_URL, EVAL_DIR)

# each line is comma separated
# first word is the COCO category
NODE_WORDS = []
WORD_TO_NODE = {}
SYNONYMS = {}

with open(SYN_PATH) as fh:
    for line in fh:
        if line.strip() == '':
            continue

        group = []

        for w in line.split(','):
            # original file has stray spaces and one capital
            w = w.strip().lower()

            if w != '' and w not in group:
                group.append(w)

        node = group[0]
        NODE_WORDS.append(node)
        SYNONYMS[node] = group

        for w in group:
            WORD_TO_NODE[w] = node

OBJECT_WORDS = set(WORD_TO_NODE.keys())
print('categories:', len(NODE_WORDS), 'object words:', len(OBJECT_WORDS))
assert len(NODE_WORDS) == 80

# double word rules
# same as original CHAIR
DOUBLE_WORDS = {}

for w in ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light',
          'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter',
          'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket',
          'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier',
          'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'home plate',
          'train track']:

    DOUBLE_WORDS[w] = w

# baby elephant -> elephant, so it does not count as person
for w in ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']:
    DOUBLE_WORDS['baby ' + w] = w
    DOUBLE_WORDS['adult ' + w] = w

# passenger jet -> jet, so it does not count as person
for w in ['jet', 'train']:
    DOUBLE_WORDS['passenger ' + w] = w

DOUBLE_WORDS['bow tie'] = 'tie'
DOUBLE_WORDS['toilet seat'] = 'toilet'
DOUBLE_WORDS['wine glas'] = 'wine glass'

INFLECT = inflect.engine()

def singularize(word):
    # keep vocab words as is
    # inflect breaks some (bus -> bu)
    if word in OBJECT_WORDS:
        return word

    singular = INFLECT.singular_noun(word)

    # inflect returns False when the word is already singular
    if singular:
        return singular
    else:
        return word

def caption_to_objects(caption):
    # returns object words and their categories
    # one entry per mention, in order
    tokens = nltk.word_tokenize(caption.lower())
    words = []

    for t in tokens:
        words.append(singularize(t))

    # merge double words so hot dog is not also counted as dog
    merged = []
    i = 0

    while i < len(words):
        pair = ' '.join(words[i:i + 2])

        if pair in DOUBLE_WORDS:
            merged.append(DOUBLE_WORDS[pair])
            i += 2
        else:
            merged.append(words[i])
            i += 1

    # seat of a toilet is not a chair
    if 'toilet' in merged and 'seat' in merged:
        kept = []

        for w in merged:
            if w != 'seat':
                kept.append(w)

        merged = kept

    object_words = []
    node_words = []

    for w in merged:
        if w in OBJECT_WORDS:
            object_words.append(w)
            node_words.append(WORD_TO_NODE[w])

    return object_words, node_words

def build_gt_cache(split):
    # {image_id: {'seg': [objects], 'cap': [[objects] per caption]}}
    gt = {}
    n_no_seg = 0

    for r in tqdm(index, desc=split):
        if r['split'] != split:
            continue
        cap_objects = []

        for c in r['captions']:
            _, nodes = caption_to_objects(c)
            cap_objects.append(nodes)
        seg = []

        if r['image_id'] in seg_objects:
            seg = sorted(seg_objects[r['image_id']])
        else:
            n_no_seg += 1

        gt[r['image_id']] = {'seg': seg, 'cap': cap_objects}

    path = os.path.join(EVAL_DIR, f'chair_gt_{split}.pkl')

    with open(path + '.tmp', 'wb') as fh:
        pickle.dump(gt, fh)

    os.replace(path + '.tmp', path)
    print(split, ': saved', len(gt), 'images,', n_no_seg, 'with no segmentation objects')
    return gt

print('helpers ready')

METRIC_ORDER = ['BLEU-4', 'METEOR', 'CIDEr', 'SPICE', 'CHAIRi', 'CHAIRs', 'avg_len', 'obj_recall']

# reference captions come from the index
if 'index' not in globals():
    with open(os.path.join(DATA_DIR, 'karpathy_index.json')) as fh:
        index = json.load(fh)
print('index loaded:', len(index), 'images')

def load_gt_objects(split):
    # GT objects per image = seg labels union objects in every human caption
    with open(os.path.join(EVAL_DIR, f'chair_gt_{split}.pkl'), 'rb') as fh:
        cache = pickle.load(fh)
    gt_objects = {}

    for image_id in cache:
        objs = set(cache[image_id]['seg'])

        for cap_objs in cache[image_id]['cap']:
            for o in cap_objs:
                objs.add(o)

        gt_objects[image_id] = objs

    return gt_objects

def load_gt_captions(split):
    # karpathy reference captions per image
    gt_caps = {}
    for r in index:
        if r['split'] == split:
            gt_caps[r['image_id']] = r['captions']
    return gt_caps

def compute_language_metrics(gt_caps, pred_caps, spice=False):
    # gt_caps: {image_id: [refs]}, pred_caps: {image_id: caption}
    gts = {}
    res = {}

    for image_id in pred_caps:
        refs = []
        for c in gt_caps[image_id]:
            refs.append({'caption': c})
        gts[image_id] = refs
        res[image_id] = [{'caption': pred_caps[image_id]}]

    # standard COCO tokenisation - lowercase, strip punctuation
    print('  tokenising...')
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    scores = {}
    print('  BLEU...')
    bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    scores['BLEU-4'] = float(bleu[3])
    print('  METEOR...')
    meteor, _ = Meteor().compute_score(gts, res)
    scores['METEOR'] = float(meteor)
    print('  CIDEr...')
    cider, _ = Cider().compute_score(gts, res)
    scores['CIDEr'] = float(cider)

    if spice:
        print('  SPICE (slow)...')
        spice_score, _ = Spice().compute_score(gts, res)
        scores['SPICE'] = float(spice_score)

    return scores

def compute_chair(pred_caps, gt_objects):
    # returns summary scores and one record per image
    per_image = []
    total_mentions = 0
    total_halluc = 0
    n_caps_halluc = 0
    total_words = 0
    recall_sum = 0.0
    n_recall = 0

    for image_id in pred_caps:
        caption = pred_caps[image_id]
        gt = gt_objects[image_id]
        _, mentioned = caption_to_objects(caption)

        # a mention is hallucinated if its category is not in the GT set
        halluc = []

        for o in mentioned:
            if o not in gt:
                halluc.append(o)

        total_mentions += len(mentioned)
        total_halluc += len(halluc)
        if len(halluc) > 0:
            n_caps_halluc += 1
        total_words += len(caption.split())

        # recall for one image = distinct GT objects mentioned / GT objects
        if len(gt) > 0:
            hit = 0
            mentioned_set = set(mentioned)

            for o in gt:
                if o in mentioned_set:
                    hit += 1

            recall_sum += hit / len(gt)
            n_recall += 1

        per_image.append({
            'image_id': image_id,
            'caption': caption,
            'mentioned_objects': mentioned,
            'hallucinated_objects': halluc,
            'gt_objects': sorted(gt)
        })

    n = len(pred_caps)
    scores = {}

    # CHAIRi: let h_k = 1 if mention k is hallucinated, else 0, over all K mentions pooled across captions
    # CHAIRi = (1/K) * sum_k h_k = hallucinated mentions / all mentions
    if total_mentions > 0:
        scores['CHAIRi'] = total_halluc / total_mentions
    else:
        scores['CHAIRi'] = 0.0

    # CHAIRs: let s_j = 1 if caption j has at least one hallucinated mention.
    # CHAIRs = (1/N) * sum_j s_j = hallucinating captions / all captions
    scores['CHAIRs'] = n_caps_halluc / n

    # avg_len = (sum_j words in caption j) / N
    scores['avg_len'] = total_words / n

    # obj_recall: r_j = |mentioned_j intersect GT_j| / |GT_j| for images with
    # |GT_j| > 0, then obj_recall = mean of r_j over those M images
    if n_recall > 0:
        scores['obj_recall'] = recall_sum / n_recall
    else:
        scores['obj_recall'] = 0.0

    scores['n_mentions'] = total_mentions
    scores['n_hallucinated'] = total_halluc
    return scores, per_image

def write_json(obj, path):
    with open(path + '.tmp', 'w') as fh:
        json.dump(obj, fh)
    os.replace(path + '.tmp', path)

def print_summary(result):
    print('-' * 40)
    print('file :', result['caption_file'])
    print('split:', result['split'], ' images:', result['n_images'])
    print('-' * 40)

    for m in METRIC_ORDER:
        if m in result['scores']:
            print(f'{m:<12}{result["scores"][m]:>10.4f}')
    print('-' * 40)

def evaluate(caption_file, split, spice=False):
    name = os.path.basename(caption_file)

    if name.endswith('.json'):
        name = name[:-5]

    result_path = os.path.join(RESULTS_DIR, name + '.eval.json')
    per_image_path = os.path.join(RESULTS_DIR, name + '.per_image.json')

    # reuse cached result unless SPICE is newly requested
    if os.path.exists(result_path):
        with open(result_path) as fh:
            cached = json.load(fh)

        if (not spice) or ('SPICE' in cached['scores']):
            print('cached result found, skipping recompute')
            print_summary(cached)
            return cached

    print('evaluating', name, 'on', split)
    with open(caption_file) as fh:
        preds = json.load(fh)

    # every image in the split exactly once, nothing extra
    gt_caps = load_gt_captions(split)
    pred_caps = {}

    for p in preds:
        image_id = int(p['image_id'])
        assert image_id not in pred_caps, f'duplicate image_id {image_id}'
        assert image_id in gt_caps, f'image_id {image_id} not in {split}'
        pred_caps[image_id] = p['caption']

    assert len(pred_caps) == len(gt_caps), \
        f'{len(pred_caps)} captions but {split} has {len(gt_caps)} images'

    print('  coverage ok:', len(pred_caps), 'images')

    scores = compute_language_metrics(gt_caps, pred_caps, spice)

    print('  CHAIR...')
    gt_objects = load_gt_objects(split)
    chair_scores, per_image = compute_chair(pred_caps, gt_objects)

    for k in chair_scores:
        scores[k] = chair_scores[k]

    result = {
        'caption_file': name,
        'split': split,
        'n_images': len(pred_caps),
        'spice': spice,
        'scores': scores
    }

    write_json(per_image, per_image_path)
    write_json(result, result_path)
    print('  saved', os.path.basename(result_path), 'and', os.path.basename(per_image_path))
    print_summary(result)
    return result

def aggregate_seeds(results):
    # mean and population std per metric across seed results
    agg = {}

    for m in METRIC_ORDER:
        values = []

        for r in results:
            if m in r['scores']:
                values.append(r['scores'][m])

        if len(values) == len(results) and len(values) > 0:
            arr = np.array(values)
            agg[m] = {'mean': float(arr.mean()), 'std': float(arr.std(ddof=0))}
    return agg

def paired_bootstrap(per_image_a, per_image_b, metric, n=1000, seed=42):
    # pair records by image id so A and B are compared on the same images
    b_by_id = {}

    for rec in per_image_b:
        b_by_id[rec['image_id']] = rec

    mention_a = []
    halluc_a = []
    mention_b = []
    halluc_b = []

    for rec in per_image_a:
        other = b_by_id[rec['image_id']]
        mention_a.append(len(rec['mentioned_objects']))
        halluc_a.append(len(rec['hallucinated_objects']))
        mention_b.append(len(other['mentioned_objects']))
        halluc_b.append(len(other['hallucinated_objects']))

    mention_a = np.array(mention_a)
    halluc_a = np.array(halluc_a)
    mention_b = np.array(mention_b)
    halluc_b = np.array(halluc_b)

    def score(halluc, mention, idx):
        # CHAIRi - pools mentions: sum h / sum m over the sampled images
        if metric == 'chair_i':
            total = mention[idx].sum()

            if total == 0:
                return 0.0

            return halluc[idx].sum() / total

        # CHAIRs - fraction of sampled captions with at least one hallucination
        return (halluc[idx] > 0).mean()

    assert metric in ('chair_i', 'chair_s'), metric
    n_img = len(mention_a)
    all_idx = np.arange(n_img)
    observed = score(halluc_b, mention_b, all_idx) - score(halluc_a, mention_a, all_idx)

    # resample images with replacement - same sample for A and B (paired).
    # diff_t = metric_B(sample t) - metric_A(sample t); the 2.5th and 97.5th
    # percentiles of diff_1..diff_n give the 95% confidence interval
    rng = np.random.default_rng(seed)
    diffs = np.zeros(n)

    for t in range(n):
        idx = rng.integers(0, n_img, n_img)
        diffs[t] = score(halluc_b, mention_b, idx) - score(halluc_a, mention_a, idx)

    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    return float(observed), lo, hi

print('evaluation helpers ready')

"""## Track 3 helpers

Same code as `phase_1/track_3/track_3.py`. `BASELINE = globals()`: `train_object_swap_run` swaps `caption_loss`, `collate`,
`RUN_KEYS` and `amp_dtype` above for the length of one run, so Track 2's `train_run` trains with the ranking term.
Object extraction is `chair_caption_objects` (same logic as Track 1's `caption_to_objects`, which is already defined above).
"""

import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

BASELINE = globals()  # namespace train_run looks its helpers up in


# ---------------------------------------------------------------------------
# Track 1 artifacts
# ---------------------------------------------------------------------------

def load_object_data(path: str | Path) -> dict:
    """Load and validate Track 1's coco_objects.json."""
    with open(path) as handle:
        data = json.load(handle)
    required = {'categories', 'synonyms', 'word_to_node', 'double_words'}
    missing = required.difference(data)
    if missing:
        raise ValueError(f'object data is missing keys: {sorted(missing)}')
    return data


def load_chair_gt(path: str | Path) -> dict:
    """Load Track 1's chair_gt_<split>.pkl cache."""
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def true_objects(cache_entry: Mapping) -> set[str]:
    """Return segmentation objects unioned with every human-caption object."""
    objects = set(cache_entry.get('seg', []))
    for caption_objects in cache_entry.get('cap', []):
        objects.update(caption_objects)
    return objects


# ---------------------------------------------------------------------------
# Co-occurrence matrix
# ---------------------------------------------------------------------------

def build_cooccurrence_matrix(
    chair_gt: Mapping[int, Mapping],
    categories: Sequence[str],
) -> tuple[np.ndarray, dict[str, int]]:
    """Count image-level object co-occurrences, including the diagonal.

    matrix[i, j] is the number of images whose CHAIR object union contains both
    categories. Unknown cache labels are ignored so a partially updated artifact
    cannot silently change matrix dimensions.
    """
    category_to_index = {category: i for i, category in enumerate(categories)}
    matrix = np.zeros((len(categories), len(categories)), dtype=np.int64)
    for entry in chair_gt.values():
        present = sorted(
            category_to_index[category]
            for category in true_objects(entry)
            if category in category_to_index
        )
        for row in present:
            matrix[row, present] += 1
    return matrix, category_to_index


def save_cooccurrence_matrix(
    path: str | Path,
    matrix: np.ndarray,
    categories: Sequence[str],
) -> None:
    """Save a portable matrix artifact with its category ordering."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, 'wb') as handle:
        pickle.dump({'categories': list(categories), 'matrix': matrix}, handle)


# ---------------------------------------------------------------------------
# CHAIR-compatible object extraction (mirrors Track 1's caption_to_objects)
# ---------------------------------------------------------------------------
_INFLECT = inflect.engine()


def _singularize(word: str, object_words: set[str]) -> str:
    """Singularize, keeping vocabulary words as-is (inflect breaks bus -> bu)."""
    if word in object_words:
        return word
    singular = _INFLECT.singular_noun(word)
    return singular if singular else word


def chair_caption_objects(caption: str, object_data: Mapping) -> tuple[list[str], list[str]]:
    """CHAIR-compatible object extraction (same logic as Track 1's caption_to_objects,
    renamed so both can live in one notebook).

    Returns (object_words, node_words) where each entry is one mention in order.
    """
    word_to_node = object_data['word_to_node']
    double_words = object_data['double_words']
    object_words = set(word_to_node.keys())

    tokens = nltk.word_tokenize(caption.lower())
    words = [_singularize(t, object_words) for t in tokens]

    merged = []
    i = 0
    while i < len(words):
        if i + 1 < len(words):
            pair = words[i] + ' ' + words[i + 1]
            if pair in double_words:
                merged.append(double_words[pair])
                i += 2
                continue
        merged.append(words[i])
        i += 1

    if 'toilet' in merged and 'seat' in merged:
        merged = [w for w in merged if w != 'seat']

    obj_words, node_words = [], []
    for w in merged:
        if w in object_words:
            obj_words.append(w)
            node_words.append(word_to_node[w])
    return obj_words, node_words


# ---------------------------------------------------------------------------
# Span-based object finding for text replacement
# ---------------------------------------------------------------------------

def build_replacement_patterns(
    object_data: Mapping,
) -> list[tuple[re.Pattern, str]]:
    """Precompute regex patterns for finding object mentions and their spans.

    Sorted longest-first so multi-word phrases ("hot dog") match before their
    sub-words ("dog"). Each synonym word also gets a plural pattern so that
    "dogs" is found even though the vocabulary only lists "dog".
    """
    word_to_node = object_data['word_to_node']
    double_words = object_data['double_words']
    entries: dict[str, str] = {}
    for word, category in word_to_node.items():
        if word not in entries:
            entries[word] = category
        plural = _INFLECT.plural_noun(word)
        if plural and plural not in entries:
            entries[plural] = category

    # Add CHAIR's multi-word rules before the individual word patterns. This
    # prevents a phrase such as "passenger jet" from matching "passenger"
    # and leaving the second word behind.
    for phrase, canonical in double_words.items():
        category = word_to_node.get(canonical)
        if category is not None:
            entries[phrase] = category

    ordered = sorted(
        entries.items(),
        key=lambda pair: (len(pair[0].split()), len(pair[0])),
        reverse=True,
    )
    return [
        (
            re.compile(
                r'(?<![A-Za-z])' + re.escape(form) + r'(?![A-Za-z])',
                re.IGNORECASE,
            ),
            category,
        )
        for form, category in ordered
    ]


def _object_mentions(
    caption: str,
    patterns: Sequence[tuple[re.Pattern, str]],
) -> list[tuple[str, str, int, int]]:
    """Find non-overlapping synonym spans as (matched_text, category, start, end)."""
    occupied: list[tuple[int, int]] = []
    mentions: list[tuple[str, str, int, int]] = []
    for pattern, category in patterns:
        for match in pattern.finditer(caption):
            start, end = match.span()
            if any(start < oe and os_ < end for os_, oe in occupied):
                continue
            occupied.append((start, end))
            mentions.append((match.group(0), category, start, end))
    return sorted(mentions, key=lambda m: m[2])


# ---------------------------------------------------------------------------
# Grammar-aware replacement
# ---------------------------------------------------------------------------

def _starts_with_vowel_sound(word: str) -> bool:
    """Rough heuristic for a/an."""
    return word[:1].lower() in 'aeiou'


def _apply_replacement(
    caption: str, start: int, end: int, source_text: str, target: str,
) -> str:
    """Replace source at [start:end] with target, preserving case and fixing a/an."""
    if source_text[:1].isupper():
        replacement = target[:1].upper() + target[1:]
    else:
        replacement = target

    prefix = caption[:start]
    art_match = re.search(r'\b(an?|An?)\s+$', prefix)
    if art_match:
        art_start = art_match.start(1)
        old_art = art_match.group(1)
        if _starts_with_vowel_sound(replacement):
            new_art = 'An' if old_art[0].isupper() else 'an'
        else:
            new_art = 'A' if old_art[0].isupper() else 'a'
        return (
            caption[:art_start] + new_art
            + caption[art_match.end(1):start] + replacement + caption[end:]
        )
    return caption[:start] + replacement + caption[end:]


def _is_plural_form(text: str) -> bool:
    """Return whether the final word in a matched object mention is plural."""
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return False
    return bool(_INFLECT.singular_noun(words[-1]))


def _target_form(target: str, source_text: str) -> str:
    """Preserve the source object's number in the replacement."""
    if not _is_plural_form(source_text):
        return target
    return _INFLECT.plural_noun(target) or target


def _has_obvious_grammar_error(caption: str) -> bool:
    """Catch simple replacement artifacts for the spot-check report."""
    text = caption.lower()
    return bool(re.search(r'\b(?:a|an)\s+(?:a|an)\b', text) or re.search(r'\s{2,}', caption))


# ---------------------------------------------------------------------------
# Swap candidate selection
# ---------------------------------------------------------------------------

def _candidate_categories(
    source: str,
    image_objects: set[str],
    categories: Sequence[str],
    matrix: np.ndarray | None,
    category_to_index: Mapping[str, int] | None,
    random_swap: bool,
    rng: random.Random,
    top_k: int = 10,
) -> list[str]:
    """Return replacement candidates.

    Structured: sorted by descending co-occurrence with source (high
    co-occurrence = plausible hallucination = hard negative), then a random
    pick from the top-k for variety.
    Random ablation: uniformly shuffled.
    """
    candidates = [
        category for category in categories
        if category != source and category not in image_objects
    ]
    if random_swap or matrix is None or category_to_index is None:
        rng.shuffle(candidates)
        return candidates
    if source not in category_to_index:
        rng.shuffle(candidates)
        return candidates

    source_row = matrix[category_to_index[source]]
    ranked = sorted(
        candidates,
        key=lambda c: (-source_row[category_to_index[c]], c),
    )
    pool = ranked[:min(top_k, len(ranked))]
    rng.shuffle(pool)
    return pool + ranked[len(pool):]


# ---------------------------------------------------------------------------
# Swap generator
# ---------------------------------------------------------------------------

def generate_swap(
    caption: str,
    image_objects: set[str],
    object_data: Mapping,
    patterns: Sequence[tuple[re.Pattern, str]],
    matrix: np.ndarray | None = None,
    category_to_index: Mapping[str, int] | None = None,
    *,
    random_swap: bool = False,
    rng: random.Random | None = None,
    top_k: int = 10,
) -> dict | None:
    """Generate one negative by replacing an object mention with an absent category.

    Uses precomputed patterns for span finding (including plural forms),
    fixes grammar (a/an, capitalisation), and validates with caption_to_objects
    that CHAIR will actually detect the injected target. If the first
    candidate collides with surrounding text (e.g. inserting "bear" next to
    "teddy" creates "teddy bear"), the next candidate is tried automatically.
    """
    rng = rng or random.Random()
    mentions = _object_mentions(caption, patterns)
    if not mentions:
        return None

    # try mentions in random order so retries explore different positions
    mention_indices = list(range(len(mentions)))
    rng.shuffle(mention_indices)

    for mi in mention_indices:
        source_text, source, start, end = mentions[mi]
        candidates = _candidate_categories(
            source, image_objects, object_data['categories'],
            matrix, category_to_index, random_swap, rng, top_k,
        )
        for target in candidates:
            target_text = _target_form(target, source_text)
            new_caption = _apply_replacement(caption, start, end, source_text, target_text)
            _, neg_objects = chair_caption_objects(new_caption, object_data)
            if target in neg_objects:
                return {
                    'caption': new_caption,
                    'source': source,
                    'target': target,
                    'target_text': target_text,
                    'source_text': source_text,
                    'random_swap': random_swap,
                }
    return None


# ---------------------------------------------------------------------------
# Spot-check negatives
# ---------------------------------------------------------------------------

def spot_check_swaps(
    examples: Sequence[tuple[str, set[str]]],
    object_data: Mapping,
    patterns: Sequence[tuple[re.Pattern, str]],
    matrix: np.ndarray,
    category_to_index: Mapping[str, int],
    *,
    n: int = 100,
    seed: int = 0,
    random_swap: bool = False,
    verbose: int = 5,
) -> dict:
    """Generate n actual negatives and validate each with caption_to_objects.

    Keeps sampling until n negatives are produced (not n inputs).
    Prints the first verbose examples for manual inspection.
    """
    rng = random.Random(seed)
    pool = list(examples)
    rng.shuffle(pool)

    checked = 0
    tried = 0
    failures = []
    printed = 0
    errors = {
        'ungrammatical': 0,
        'target_not_absent_from_image': 0,
        'target_not_detected_by_chair': 0,
        'plural_not_preserved': 0,
    }

    for caption, image_objects in pool:
        if checked >= n:
            break
        tried += 1
        result = generate_swap(
            caption, image_objects, object_data, patterns,
            matrix, category_to_index,
            random_swap=random_swap, rng=rng,
        )
        if result is None:
            continue
        checked += 1

        # validate with CHAIR-compatible extraction
        _, neg_objects = chair_caption_objects(result['caption'], object_data)
        target_found = result['target'] in neg_objects
        target_absent = result['target'] not in image_objects
        source_same = result['source'] == result['target']
        ungrammatical = _has_obvious_grammar_error(result['caption'])
        plural_mismatch = _is_plural_form(result['source_text']) != _is_plural_form(result['target_text'])

        ok = target_found and target_absent and not source_same
        if ungrammatical:
            errors['ungrammatical'] += 1
        if not target_absent:
            errors['target_not_absent_from_image'] += 1
        if not target_found:
            errors['target_not_detected_by_chair'] += 1
        if plural_mismatch:
            errors['plural_not_preserved'] += 1
        if not ok:
            failures.append({
                **result,
                'target_in_neg_caption': target_found,
                'target_in_image': not target_absent,
                'ungrammatical': ungrammatical,
                'plural_mismatch': plural_mismatch,
            })

        if printed < verbose:
            tag = 'OK' if ok else 'FAIL'
            print(f'  [{tag}] "{caption}"')
            print(f'       -> "{result["caption"]}"')
            print(f'       {result["source"]} -> {result["target"]}')
            printed += 1

    rates = {name: count / checked if checked else 0.0 for name, count in errors.items()}
    return {
        'requested': n,
        'tried': tried,
        'checked': checked,
        'failures': failures,
        'errors': errors,
        'error_rates': rates,
    }


# ---------------------------------------------------------------------------
# One fixed negative per training caption
# ---------------------------------------------------------------------------

def build_negatives(
    captions: Sequence[Mapping],
    chair_gt: Mapping[int, Mapping],
    object_data: Mapping,
    patterns: Sequence[tuple[re.Pattern, str]],
    matrix: np.ndarray,
    category_to_index: Mapping[str, int],
    *,
    random_swap: bool = False,
    seed: int = 0,
    cache_path: str | None = None,
) -> list[str | None]:
    """One negative caption per entry of split_data['captions'] (None if no valid swap).

    The swap seed is separate from the training seed, so every training seed (and
    lambda) sees the same negatives and runs differ only in the loss. Slow on the
    full train split (regex + CHAIR validation per caption), so cache it.
    """
    texts = [c['caption'] for c in captions]
    settings = {'random_swap': random_swap, 'seed': seed}

    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, 'rb') as fh:
            cached = pickle.load(fh)
        if cached['texts'] == texts and cached['settings'] == settings:
            return cached['negatives']
        print('negative cache does not match captions/settings, rebuilding')

    rng = random.Random(seed)
    negatives = []
    for c in tqdm(captions, desc='random swaps' if random_swap else 'structured swaps'):
        entry = chair_gt.get(int(c['image_id']))
        if entry is None:
            negatives.append(None)
            continue
        result = generate_swap(
            c['caption'], true_objects(entry), object_data, patterns,
            matrix, category_to_index, random_swap=random_swap, rng=rng,
        )
        negatives.append(None if result is None else result['caption'])

    if cache_path is not None:
        with open(cache_path + '.tmp', 'wb') as fh:
            pickle.dump({'texts': texts, 'settings': settings, 'negatives': negatives}, fh)
        os.replace(cache_path + '.tmp', cache_path)
    return negatives


def tokenize_negatives(negatives: Sequence[str | None], tokenizer) -> list[list[int] | None]:
    # GPT-2 ids, no EOS (same as track_2.tokenize_captions); None stays None
    present = [i for i, n in enumerate(negatives) if n is not None]
    ids = tokenizer([negatives[i] for i in present], add_special_tokens=False)['input_ids']
    out: list[list[int] | None] = [None] * len(negatives)
    for i, t in zip(present, ids):
        out[i] = t
    return out


class ObjectSwapDataset(CaptionDataset):
    # one item per caption: (token ids, feature row, negative token ids or None)
    def __init__(self, split_data, token_lists, max_len, negative_token_lists, swap):
        super().__init__(split_data, token_lists, max_len)
        assert len(negative_token_lists) == len(self.tokens)
        assert swap in ('structured', 'random'), swap
        self.swap = swap
        self.negatives = []
        for pos, neg in zip(self.tokens, negative_token_lists):
            neg = None if neg is None else neg[:max_len]
            # truncation can cut the swapped object off; then the pair carries no signal
            self.negatives.append(None if neg is None or neg == pos else neg)

    def __getitem__(self, i):
        return self.tokens[i], self.rows[i], self.negatives[i]

    def negative_rate(self):
        return sum(n is not None for n in self.negatives) / len(self.negatives)


def pack_swap_batch(batch, collate=collate):
    """[(pos, row, neg)] -> tokens/labels [2B, L] (positives, then negatives), rows [B].

    Rows with no negative reuse the positive tokens with all labels -100, so they
    are padded like the rest but count for nothing.
    """
    n = len(batch)
    seqs = [pos for pos, _, _ in batch] + [pos if neg is None else neg for pos, _, neg in batch]
    tokens, labels, _ = collate([(s, 0) for s in seqs])
    for i, (_, _, neg) in enumerate(batch):
        if neg is None:
            labels[n + i] = -100
    rows = torch.tensor([r for _, r, _ in batch], dtype=torch.long)
    return tokens, labels, rows


# ---------------------------------------------------------------------------
# Loss (batched)
# ---------------------------------------------------------------------------

def object_swap_loss(model, tokens, labels, features, *, ranking_weight=1.0, margin=0.2):
    """Positives and negatives in one GPT-2 forward pass, sharing one mapping-network pass.

    tokens/labels: [2B, L] from pack_swap_batch; features: [B, 640] raw CLIP.
    The GPT-2 part mirrors Track 2's caption_loss line for line, so 'nll' equals
    caption_loss on the positives (checked in smoke_test_loss).
    """
    n = features.shape[0]
    assert tokens.shape[0] == 2 * n and labels.shape == tokens.shape, (tokens.shape, features.shape)

    prefix_embeds = prefix_embeddings(model, features)
    prefix_embeds = torch.cat((prefix_embeds, prefix_embeds), dim=0)
    text_embeds = model.gpt.transformer.wte(tokens)
    inputs = torch.cat((prefix_embeds.to(text_embeds.dtype), text_embeds), dim=1)

    hidden = model.gpt.transformer(inputs_embeds=inputs).last_hidden_state
    p = model.prefix_length
    logits = model.gpt.lm_head(hidden[:, p - 1:-1])
    token_nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                ignore_index=-100, reduction='none').reshape(labels.shape)

    counts = (labels != -100).sum(dim=1)
    nll = token_nll[:n].sum() / counts[:n].sum()          # == caption_loss(positives)
    per_caption = token_nll.sum(dim=1) / counts.clamp_min(1)
    positive, negative = per_caption[:n], per_caption[n:]
    has_negative = (counts[n:] > 0).float()
    n_negatives = has_negative.sum()

    hinge = F.relu(margin + positive - negative) * has_negative
    ranking = hinge.sum() / n_negatives.clamp_min(1)
    return {
        'loss': nll + ranking_weight * ranking,
        'nll': nll,
        'ranking_loss': ranking,
        'positive_nll': (positive * has_negative).sum() / n_negatives.clamp_min(1),
        'negative_nll': (negative * has_negative).sum() / n_negatives.clamp_min(1),
        'n_negatives': n_negatives,
    }


# ---------------------------------------------------------------------------
# Training: Track 2's train_run with the ranking term added
# ---------------------------------------------------------------------------

SWAP_RUN_KEYS = RUN_KEYS + ['ranking_weight', 'margin', 'swap']


def object_swap_config(ranking_weight, margin=0.2, swap='structured', **overrides):
    # Track 2 defaults (the baseline's settings) plus the three Track 3 knobs
    assert swap in ('structured', 'random'), swap
    cfg = default_config(**overrides)
    cfg.update({'ranking_weight': float(ranking_weight), 'margin': float(margin), 'swap': swap})
    return cfg


def run_name(cfg):
    return (f"objswap_{cfg['swap']}_lam{cfg['ranking_weight']:g}_m{cfg['margin']:g}_"
            f"{config_name(cfg)}_seed{cfg['seed']}")


@contextmanager
def patched_baseline(ranking_weight, margin, ns=None):
    """Make Track 2's train_run / dataset_loss / benchmark (in namespace ns) use the ranking term.

    - collate: ObjectSwapDataset items -> pack_swap_batch; plain (tokens, row) items
      (the val set) -> the original collate. With ranking_weight 0 the negatives
      are dropped, so batches are exactly the baseline's.
    - caption_loss: a packed [2B] batch -> object_swap_loss; anything else (val loss,
      ranking_weight 0) -> the original caption_loss, so val_loss stays comparable.
    - RUN_KEYS: resume also checks ranking_weight / margin / swap.
    - amp_dtype: fp16, the precision all baseline seeds used (as in train_baseline.py).
    """
    ns = BASELINE if ns is None else ns
    names = ('caption_loss', 'collate', 'RUN_KEYS', 'amp_dtype')
    saved = {k: ns[k] for k in names}
    base_loss, base_collate = saved['caption_loss'], saved['collate']
    assert getattr(base_loss, '_track3', False) is False, 'Track 2 helpers are already patched'

    def collate(batch):
        if len(batch[0]) == 2 or ranking_weight == 0:
            return base_collate([(t, r) for t, r, *_ in batch])
        return pack_swap_batch(batch, base_collate)

    def caption_loss(model, tokens, labels, features, reduction='mean'):
        if tokens.shape[0] == features.shape[0]:
            return base_loss(model, tokens, labels, features, reduction)
        assert reduction == 'mean', reduction
        return object_swap_loss(model, tokens, labels, features,
                                ranking_weight=ranking_weight, margin=margin)['loss']

    caption_loss._track3 = True
    ns.update(caption_loss=caption_loss, collate=collate, RUN_KEYS=SWAP_RUN_KEYS,
              amp_dtype=lambda device: torch.float16)
    try:
        yield
    finally:
        ns.update(saved)


def train_object_swap_run(cfg, train_set, val_set, run_dir, device, **kwargs):
    """Track 2's train_run with the ranking loss; same resume/deadline/max_steps behaviour.

    val_set is a plain CaptionDataset; its val_loss is the plain caption NLL.
    """
    assert isinstance(train_set, ObjectSwapDataset), 'train_set must be an ObjectSwapDataset'
    assert train_set.swap == cfg['swap'], (train_set.swap, cfg['swap'])
    # train_run returns a finished run without looking at its config, so check here
    log_path = os.path.join(run_dir, 'run_log.json')
    if os.path.exists(log_path):
        done_cfg = read_json(log_path)['config']
        for k in SWAP_RUN_KEYS:
            assert done_cfg.get(k) == cfg[k], f'{run_dir} was trained with a different {k}'

    with patched_baseline(cfg['ranking_weight'], cfg['margin']):
        return BASELINE['train_run'](cfg, train_set, val_set, run_dir, device, **kwargs)


# ---------------------------------------------------------------------------
# Smoke test with pretrained ClipCap
# ---------------------------------------------------------------------------

def smoke_test_loss(
    model, tokenizer, val_data, chair_gt_val, object_data, patterns,
    matrix, category_to_index, n=40,
) -> dict[str, float]:
    """Check the batched loss on real val captions paired to their own features.

    val_data comes from Track 2's load_features.

    1. 'nll' from the batched pos+neg pass equals Track 2's caption_loss on the positives.
    2. With ranking_weight 0 the patched collate/loss are exactly the baseline's.
    3. Real captions get lower NLL than their object swaps.
    4. The mapping receives gradients while frozen GPT-2 does not.
    5. A batch with no negatives remains finite.
    """
    device = next(model.parameters()).device
    rng = random.Random(0)
    batch, texts = [], []

    for cap_entry in val_data['captions']:
        if len(batch) >= n:
            break
        entry = chair_gt_val.get(int(cap_entry['image_id']))
        if entry is None:
            continue
        result = generate_swap(
            cap_entry['caption'], true_objects(entry), object_data, patterns,
            matrix, category_to_index, rng=rng,
        )
        if result is None:
            continue
        pos = tokenizer.encode(cap_entry['caption'], add_special_tokens=False)
        neg = tokenizer.encode(result['caption'], add_special_tokens=False)
        batch.append((pos, cap_entry['clip_embedding'], neg))
        texts.append((cap_entry['caption'], result['caption']))

    assert len(batch) == n, f'need {n} swappable captions, got {len(batch)}'
    features = torch.as_tensor(val_data['features'])[[r for _, r, _ in batch]].to(device)
    tokens, labels, _ = pack_swap_batch(batch)
    base_tokens, base_labels, _ = collate([(pos, r) for pos, r, _ in batch])

    model.eval()
    model.zero_grad(set_to_none=True)
    parts = object_swap_loss(model, tokens.to(device), labels.to(device), features)
    with torch.no_grad():
        reference = BASELINE['caption_loss'](model, base_tokens.to(device), base_labels.to(device), features)
        with patched_baseline(0.0, 0.2):
            t0, l0, _ = BASELINE['collate'](batch)
            loss0 = BASELINE['caption_loss'](model, t0.to(device), l0.to(device), features)
        none_tokens, none_labels, _ = pack_swap_batch([(pos, r, None) for pos, r, _ in batch])
        none_parts = object_swap_loss(model, none_tokens.to(device), none_labels.to(device), features)

    parts['loss'].backward()
    mapping_grad = max(
        float(parameter.grad.abs().max())
        for parameter in model.clip_project.parameters()
        if parameter.grad is not None
    )
    gpt_grads = [parameter.grad for parameter in model.gpt.parameters()]
    assert mapping_grad > 0.0, 'ranking loss did not reach the mapping network'
    assert all(gradient is None for gradient in gpt_grads), 'GPT-2 received gradients'
    model.zero_grad(set_to_none=True)

    out = {name: float(value.detach().cpu()) for name, value in parts.items()}
    out['caption_loss'] = float(reference)
    out['mapping_grad_max'] = mapping_grad
    out['none_negative_loss'] = float(none_parts['loss'])

    for i in range(min(n, 3)):
        print(f'  [{i}] pos: "{texts[i][0]}"')
        print(f'       neg: "{texts[i][1]}"')
    print(f"  nll={out['nll']:.4f}  caption_loss={out['caption_loss']:.4f}")
    print(f"  positive_nll={out['positive_nll']:.3f}  negative_nll={out['negative_nll']:.3f}")
    print(f"  ranking_loss={out['ranking_loss']:.3f}  total_loss={out['loss']:.3f}")

    assert all(np.isfinite(v) for v in out.values()), out
    assert abs(out['nll'] - out['caption_loss']) < 1e-4, 'batched positive NLL differs from caption_loss'
    assert torch.equal(t0, base_tokens) and torch.equal(l0, base_labels), 'lambda=0 batches differ from baseline'
    assert abs(float(loss0) - out['caption_loss']) < 1e-6, 'lambda=0 loss differs from caption_loss'
    assert out['positive_nll'] < out['negative_nll'], 'real captions should have lower NLL than swaps'
    assert np.isfinite(out['none_negative_loss']), 'all-none negative batch is not finite'
    assert float(none_parts['n_negatives']) == 0.0, 'all-none batch unexpectedly has negatives'
    return out


# ---------------------------------------------------------------------------
# Paired bootstrap against the baseline, seed by seed
# ---------------------------------------------------------------------------

def _chair_from_per_image(records):
    mentions = sum(len(r['mentioned_objects']) for r in records)
    halluc = [len(r['hallucinated_objects']) for r in records]
    return {'chair_i': sum(halluc) / mentions if mentions else 0.0,
            'chair_s': sum(h > 0 for h in halluc) / len(records)}


def compare_to_baseline(baseline_files, track3_files, paired_bootstrap,
                        metrics=('chair_i', 'chair_s'), n=1000, seed=42):
    """Track 1's paired_bootstrap on .per_image.json files, baseline seed s vs Track 3 seed s.

    baseline_files, track3_files: {seed: path}. paired_bootstrap is Track 1's function
    (track1_eval.paired_bootstrap / the notebook copy). diff = Track 3 - baseline, so negative = fewer
    hallucinations; 'significant' means the 95% CI excludes 0.
    """
    assert set(baseline_files) == set(track3_files), 'compare the same seeds'
    rows = []
    for s in sorted(baseline_files):
        base = read_json(baseline_files[s])
        ours = read_json(track3_files[s])
        assert {r['image_id'] for r in base} == {r['image_id'] for r in ours}, f'seed {s}: image sets differ'
        base_scores, our_scores = _chair_from_per_image(base), _chair_from_per_image(ours)
        for m in metrics:
            diff, lo, hi = paired_bootstrap(base, ours, m, n=n, seed=seed)
            rows.append({'seed': s, 'metric': m, 'baseline': base_scores[m], 'track3': our_scores[m],
                         'diff': diff, 'ci_lo': lo, 'ci_hi': hi, 'significant': hi < 0 or lo > 0})
    for r in rows:
        print(f"seed {r['seed']} {r['metric']}: baseline {100 * r['baseline']:.2f}  track3 {100 * r['track3']:.2f}  "
              f"diff {100 * r['diff']:+.2f} [{100 * r['ci_lo']:+.2f}, {100 * r['ci_hi']:+.2f}]"
              + ('  *' if r['significant'] else ''))
    return rows

"""# Object list and co-occurrence matrix
"""

object_data = load_object_data(os.path.join(TRACK1_ROOT, 'eval', 'coco_objects.json'))
chair_gt = load_chair_gt(os.path.join(TRACK1_ROOT, 'eval', 'chair_gt_train.pkl'))
categories = object_data['categories']
cooccurrence, category_to_index = build_cooccurrence_matrix(chair_gt, categories)

matrix_path = os.path.join(TRACK3_ROOT, 'eval/cooccurrence_train.pkl')
save_cooccurrence_matrix(matrix_path, cooccurrence, categories)
print('categories:', len(categories))
print('images:', len(chair_gt))
print('matrix:', cooccurrence.shape, cooccurrence.dtype)
print('saved:', matrix_path)
assert np.array_equal(cooccurrence, cooccurrence.T)
# each object co-occurs with itself at least as often as with any other
diag = np.diag(cooccurrence)
off_diag_max = (cooccurrence - np.diag(diag)).max(axis=1)
assert np.all(diag >= off_diag_max), 'diagonal must be >= every off-diagonal in its row'

# precompute regex patterns for span-based object finding (includes plural forms)
patterns = build_replacement_patterns(object_data)
print('replacement patterns:', len(patterns))

"""The truth for each image is `seg` union every object list in `cap`; therefore the diagonal counts images containing each object and off-diagonal entries count image-level co-occurrence.

# Swap generation and 100-negative spot checks
"""

with open(os.path.join(TRACK1_ROOT, 'data', 'karpathy_index.json')) as handle:
    karpathy_index = json.load(handle)

examples = []
for row in karpathy_index:
    if row['split'] != 'train' or row['image_id'] not in chair_gt:
        continue
    image_objects = true_objects(chair_gt[row['image_id']])
    for caption in row['captions']:
        examples.append((caption, image_objects))

print('structured swaps:')
structured_check = spot_check_swaps(
    examples, object_data, patterns, cooccurrence, category_to_index,
    n=100, seed=0, verbose=5,
)
print()
print('random swaps:')
random_check = spot_check_swaps(
    examples, object_data, patterns, cooccurrence, category_to_index,
    n=100, seed=0, random_swap=True, verbose=5,
)
print()
print('structured:', {k: v for k, v in structured_check.items() if k != 'failures'})
print('random:', {k: v for k, v in random_check.items() if k != 'failures'})
if structured_check['failures']:
    print('STRUCTURED FAILURES:', structured_check['failures'])
if random_check['failures']:
    print('RANDOM FAILURES:', random_check['failures'])
assert not structured_check['failures'], 'structured swap validation failed'
assert not random_check['failures'], 'random swap validation failed'
assert structured_check['checked'] == 100, f'only got {structured_check["checked"]} structured swaps'
assert random_check['checked'] == 100, f'only got {random_check["checked"]} random swaps'

example_caption, example_objects = examples[0]
demo_rng = random.Random(0)
structured = generate_swap(
    example_caption, example_objects, object_data, patterns,
    cooccurrence, category_to_index, rng=demo_rng,
)
random_negative = generate_swap(
    example_caption, example_objects, object_data, patterns,
    cooccurrence, category_to_index, random_swap=True, rng=demo_rng,
)
print('original:', example_caption)
print('structured:', structured)
print('random:', random_negative)

r"""# Pretrained ClipCap loss smoke test

Real val captions paired to their own CLIP features, negatives built against each image's CHAIR ground truth. Checks that
1. the batched positive NLL equals Track 2's `caption_loss` (no EOS),
2. with $\lambda = 0$ the batches and loss are exactly the baseline's,
3. real captions get lower NLL than their object swaps.
"""

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
val_data = load_features(os.path.join(FEAT_DIR, 'coco_rn50x4_val.pkl'))
chair_gt_val = load_chair_gt(os.path.join(EVAL_DIR, 'chair_gt_val.pkl'))

pretrained = build_model('transformer', 40)
load_pretrained_checkpoint(pretrained, PRETRAINED_CKPT)
pretrained = pretrained.to(device).eval()

loss_values = smoke_test_loss(
    pretrained, tokenizer, val_data, chair_gt_val, object_data, patterns,
    cooccurrence, category_to_index,
)
print(loss_values)
del pretrained
torch.cuda.empty_cache()

"""# Training data and negatives

Same train/val sets as the baseline (`tokenize_captions`, `max_seq_len`, `CaptionDataset`). Each train caption gets one fixed negative
per swap type (swap seed 0 for every training seed, so runs differ only in the loss). Building them takes a while on the full train
split; they are cached in `track_3/cache/`.
"""

train_data = load_features(os.path.join(FEAT_DIR, 'coco_rn50x4_train.pkl'))
train_tokens = tokenize_captions(train_data['captions'], tokenizer, os.path.join(CACHE_DIR, 'train_tokens.pkl'))
val_tokens = tokenize_captions(val_data['captions'], tokenizer, os.path.join(CACHE_DIR, 'val_tokens.pkl'))
MAX_LEN = max_seq_len(train_tokens)
val_set = CaptionDataset(val_data, val_tokens, MAX_LEN)
print('max caption tokens:', MAX_LEN, ' train captions:', len(train_tokens), ' val captions:', len(val_set))

train_sets = {}
for swap in ['structured', 'random']:
    negatives = build_negatives(
        train_data['captions'], chair_gt, object_data, patterns, cooccurrence, category_to_index,
        random_swap=(swap == 'random'), seed=0, cache_path=os.path.join(CACHE_DIR, f'train_negatives_{swap}.pkl'),
    )
    train_sets[swap] = ObjectSwapDataset(train_data, train_tokens, MAX_LEN,
                                         tokenize_negatives(negatives, tokenizer), swap)
    print(f'{swap}: {100 * train_sets[swap].negative_rate():.1f}% of train captions have a negative')
    i = next(i for i, n in enumerate(negatives) if n is not None)
    print('   ', train_data['captions'][i]['caption'], '->', negatives[i])

    val_negatives = build_negatives(
        val_data['captions'], chair_gt_val, object_data, patterns, cooccurrence, category_to_index,
        random_swap=(swap == 'random'), seed=0,
        cache_path=os.path.join(CACHE_DIR, f'val_negatives_{swap}.pkl'),
    )
    val_rate = sum(negative is not None for negative in val_negatives) / len(val_negatives)
    print(f'{swap}: {100 * val_rate:.1f}% of val captions have a negative')

r"""# Runs

Baseline settings (transformer, prefix 40, 10 epochs, bs 40, lr 2e-5, 5000 warmup, fp16) plus $\lambda$, margin and swap type.
`lam0` is the sanity run: it should match the baseline seed for seed (its bootstrap CIs below should cover 0).
"""

BEST_SPEC = {'mapping': 'transformer', 'prefix_length': 40}   # Track 2 baseline (results/baseline_plan.json)
EPOCHS = 10
SEEDS = [0, 1, 2]
MARGIN = 0.2
RUN_SPECS = [
    {'ranking_weight': 0.0, 'swap': 'structured'},   # sanity: reproduces the baseline
    {'ranking_weight': 1.0, 'swap': 'structured'},   # main method
    {'ranking_weight': 1.0, 'swap': 'random'},       # ablation: random swaps
]

def run_paths(spec, seed):
    cfg = object_swap_config(spec['ranking_weight'], MARGIN, spec['swap'], **BEST_SPEC, epochs=EPOCHS, seed=seed)
    name = run_name(cfg)
    return {'cfg': cfg, 'name': name, 'run_dir': os.path.join(RUNS_DIR, name),
            'captions': os.path.join(RESULTS_DIR, f'clipcap_{name}_val.json'),
            'per_image': os.path.join(RESULTS_DIR, f'clipcap_{name}_val.per_image.json'),
            'eval': os.path.join(RESULTS_DIR, f'clipcap_{name}_val.eval.json'),
            'model': os.path.join(MODELS_DIR, f'clipcap_{name}_mapping.pt')}

for spec in RUN_SPECS:
    for seed in SEEDS:
        p = run_paths(spec, seed)
        print('run:', p['name'])
        if not (os.path.exists(p['captions']) and os.path.exists(p['model'])):
            model, run_log = train_object_swap_run(p['cfg'], train_sets[spec['swap']], val_set, p['run_dir'], device)
            if not os.path.exists(p['model']):
                shutil.copy(os.path.join(p['run_dir'], 'mapping_final.pt'), p['model'])
            decode_split(model, tokenizer, val_data, p['captions'], device)
            print('  train hours:', round(run_log['train_seconds'] / 3600, 2))
            del model
            torch.cuda.empty_cache()
        evaluate(p['captions'], 'val', spice=False)

"""# Paired bootstrap vs the baseline, seed by seed

Track 1's `paired_bootstrap` (1000 resamples of the 5000 val images, same images for both models). diff = Track 3 - baseline,
so negative = fewer hallucinations; `*` = 95% CI excludes 0.
"""

baseline_files = {s: os.path.join(BASELINE_RESULTS, f'clipcap_baseline_seed{s}_val.per_image.json') for s in SEEDS}
summary = {'margin': MARGIN, 'epochs': EPOCHS, 'seeds': SEEDS, 'runs': []}

for spec in RUN_SPECS:
    paths = {s: run_paths(spec, s) for s in SEEDS}
    print(f"\n=== lambda={spec['ranking_weight']:g}, swap={spec['swap']} ===")
    rows = compare_to_baseline(baseline_files, {s: paths[s]['per_image'] for s in SEEDS}, paired_bootstrap)
    agg = aggregate_seeds([read_json(paths[s]['eval']) for s in SEEDS])
    for m in ['CIDEr', 'BLEU-4', 'CHAIRi', 'CHAIRs', 'obj_recall', 'avg_len']:
        scale = 1 if m == 'avg_len' else 100
        print(f"  {m:<11}{agg[m]['mean'] * scale:8.2f} +- {agg[m]['std'] * scale:.2f}")
    summary['runs'].append({**spec, 'aggregate': agg, 'bootstrap': rows})

write_json(summary, os.path.join(RESULTS_DIR, 'track3_summary.json'))
print('\nsaved', os.path.join(RESULTS_DIR, 'track3_summary.json'))

"""# End of Session
"""

print('Track 3 completed successfully.')
