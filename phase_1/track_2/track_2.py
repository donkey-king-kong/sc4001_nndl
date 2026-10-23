"""Track 2: ClipCap baseline - training, greedy decoding, sweep and seed runs.

Same code as the helper cells in track_2.ipynb, kept as a module so it can be
imported and tested outside Colab.

Features are raw RN50x4 CLIP vectors. Every prefix is L2-normalised inside
the model code (normalise_prefix) before the mapping network, as the
pretrained checkpoint expects. GPT-2 is frozen; only clip_project trains.
"""

import json
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

MAX_TOKENS = 30  # team rule: greedy, max 30 tokens, stop at '.'


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Greedy decoding (same algorithm as Track 1's greedy_decode_batch)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

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
