"""Phase 2, B and C: integrate and run.

ClipCap baseline (Track 2) trained with the object-swap ranking term (Track 3):
    total loss = caption NLL + lambda * ranking loss
Runs the lambda sweep, picks the best lambda on val, then the random-swap ablation and the
optional margin vs unlikelihood comparison, and compares every run with the baseline seed by seed.
Val only: nothing here reads the test split (that is Phase 3).

Code reused from Phase 1 (copied next to this file):
    track_2.py      Track 2 training / decoding (phase_1/track_2/track_2.py, unchanged)
    track1_eval.py  Track 1 scoring, CHAIR, paired bootstrap (phase_1/track_2/tc1/track1_eval.py, unchanged)
    object_swap.py  Track 3 negatives and loss (phase_1/track_3/track_3.py helpers, Phase 2 changes marked)

Stages, in order (each one can be run again; finished work is skipped):
    python run_phase2.py --stage check          # environment + loss smoke test on the pretrained checkpoint
    python run_phase2.py --stage prepare        # negatives caches, spot check, sheet for the manual review
    python run_phase2.py --stage sweep          # lambda in {0, 0.1, 0.3, 1, 3} x 3 seeds, structured swaps
    python run_phase2.py --stage select         # best lambda on val -> results/best_lambda.json
    python run_phase2.py --stage ablation       # random swaps at the best lambda x 3 seeds
    python run_phase2.py --stage unlikelihood   # optional: unlikelihood loss at the best lambda x 3 seeds
    python run_phase2.py --stage summary        # bootstrap vs baseline, aux-loss check, results/runs.csv

Splitting runs across GPUs: start one process per GPU with --worker i --n-workers k
(run j of the stage goes to worker j % k). Every process rewrites results/runs.csv, the run sheet.

Exit codes: 0 = everything requested is done, 3 = stopped at the time limit (submit again), other = error.
"""

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import time

START = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

STAGES = ['check', 'prepare', 'sweep', 'select', 'ablation', 'unlikelihood', 'summary']

parser = argparse.ArgumentParser()
parser.add_argument('--stage', required=True, choices=STAGES)
parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lambdas', type=float, nargs='+', default=[0.0, 0.1, 0.3, 1.0, 3.0], help='sweep values')
parser.add_argument('--margin', type=float, default=0.2)
parser.add_argument('--best-lambda', type=float, default=None, help='skip select and use this lambda')
parser.add_argument('--ul-lambdas', type=float, nargs='+', default=None,
                    help='lambdas for the unlikelihood runs (default: the best margin lambda)')
parser.add_argument('--max-cider-drop', type=float, default=2.0,
                    help='select: allowed drop in mean val CIDEr (points) below the baseline')
parser.add_argument('--worker', type=int, default=0)
parser.add_argument('--n-workers', type=int, default=1)
parser.add_argument('--time-limit-min', type=float, default=None, help='save and stop before this many minutes')
parser.add_argument('--device', default=None, help='default: cuda if available, else cpu')
parser.add_argument('--skip-eval', action='store_true', help='testing only: skip scoring')
parser.add_argument('--tiny', action='store_true', help='testing only: tiny data, small batches, separate output folder')
parser.add_argument('--root', default=PROJECT_ROOT, help='repo root (folder with phase_1/)')
parser.add_argument('--track1-dir', default=None, help='default: <root>/phase_1/track_1 (data/, eval/, features/, models/)')
parser.add_argument('--features-dir', default=None, help='default: <track1-dir>/features')
parser.add_argument('--baseline-results', default=None, help='default: <root>/phase_1/track_2/results')
parser.add_argument('--cooccurrence', default=None, help='default: <root>/phase_1/track_3/eval/cooccurrence_train.pkl')
parser.add_argument('--clip-root', default=None, help='ClipCap checkout (default: $CLIP_PREFIX_CAPTION_ROOT or <root>/CLIP_prefix_caption)')
parser.add_argument('--out', default=HERE, help='folder for runs/, results/, models/, cache/')
args = parser.parse_args()
assert 0 <= args.worker < args.n_workers, 'need 0 <= --worker < --n-workers'

# chosen by the Track 2 mini sweep (phase_1/track_2/results/baseline_plan.json)
BEST_SPEC = {'mapping': 'transformer', 'prefix_length': 40}
SWAP_SEED = 0          # one fixed negative per caption, the same for every training seed and lambda
EXIT_TIME_LIMIT = 3
METRICS = ['BLEU-4', 'METEOR', 'CIDEr', 'CHAIRi', 'CHAIRs', 'avg_len', 'obj_recall']

TRACK1_DIR = args.track1_dir or os.path.join(args.root, 'phase_1', 'track_1')
DATA_DIR = os.path.join(TRACK1_DIR, 'data')
EVAL_DIR = os.path.join(TRACK1_DIR, 'eval')
FEAT_DIR = args.features_dir or os.path.join(TRACK1_DIR, 'features')
PRETRAINED_CKPT = os.path.join(TRACK1_DIR, 'models', 'clipcap_transformer_coco_pretrained.pt')
BASELINE_RESULTS = args.baseline_results or os.path.join(args.root, 'phase_1', 'track_2', 'results')
COOCCURRENCE = args.cooccurrence or os.path.join(args.root, 'phase_1', 'track_3', 'eval', 'cooccurrence_train.pkl')
CLIP_ROOT = args.clip_root or os.environ.get('CLIP_PREFIX_CAPTION_ROOT', os.path.join(args.root, 'CLIP_prefix_caption'))

OUT = os.path.join(args.out, 'tiny') if args.tiny else args.out
RESULTS_DIR = os.path.join(OUT, 'results')
RUNS_DIR = os.path.join(OUT, 'runs')
MODELS_DIR = os.path.join(OUT, 'models')
CACHE_DIR = os.path.join(OUT, 'cache')
for d in [RESULTS_DIR, RUNS_DIR, MODELS_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# track1_eval reads these when imported; evaluate() writes .eval.json / .per_image.json to RESULTS_DIR
os.environ['TRACK2_DATA_DIR'] = DATA_DIR
os.environ['TRACK2_EVAL_DIR'] = EVAL_DIR
os.environ['TRACK2_RESULTS_DIR'] = RESULTS_DIR
os.environ['JAVA_TOOL_OPTIONS'] = '-Xlog:disable'  # Java warnings corrupt pycocoevalcap output

sys.path.insert(0, HERE)

import nltk  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import GPT2TokenizerFast  # noqa: E402

import track_2  # noqa: E402
from track_2 import (CaptionDataset, build_model, decode_split, load_features,  # noqa: E402
                     load_pretrained_checkpoint, max_seq_len, read_json, subset_images,
                     tokenize_captions, write_json)
import object_swap as osw  # noqa: E402

# all baseline seeds ran in fp16 (Colab T4, then TC1 V100); keep the same precision on newer GPUs
track_2.amp_dtype = lambda device: torch.float16

device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
deadline = None if args.time_limit_min is None else START + args.time_limit_min * 60


def minutes_left():
    return float('inf') if deadline is None else (deadline - time.time()) / 60


def patch_clipcap():
    # ClipCap's train.py imports AdamW from transformers, removed in newer versions; patch the line once
    train_py = os.path.join(CLIP_ROOT, 'train.py')
    if not os.path.exists(train_py):
        raise FileNotFoundError(f'ClipCap checkout not found at {CLIP_ROOT}; pass --clip-root')
    old = 'from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup'
    new = ('from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup\n'
           'from torch.optim import AdamW')
    with open(train_py) as fh:
        src = fh.read()
    if old in src:
        with open(train_py, 'w') as fh:
            fh.write(src.replace(old, new))
        print('patched ClipCap train.py AdamW import')
    if CLIP_ROOT not in sys.path:
        sys.path.append(CLIP_ROOT)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def spec(ranking_weight, swap='structured', loss_type='margin'):
    return {'ranking_weight': float(ranking_weight), 'swap': swap, 'loss_type': loss_type}


def best_lambda():
    # --best-lambda, else the select stage's choice, else None
    if args.best_lambda is not None:
        return args.best_lambda
    path = os.path.join(RESULTS_DIR, 'best_lambda.json')
    if os.path.exists(path):
        return read_json(path)['best_lambda']
    return None


def stage_specs(stage):
    if stage == 'sweep':
        return [spec(lam) for lam in args.lambdas]

    best = best_lambda()
    if stage == 'ablation':
        assert best is not None, 'no best lambda yet: run --stage select first, or pass --best-lambda'
        return [spec(best, 'random')]
    if stage == 'unlikelihood':
        lams = args.ul_lambdas
        if lams is None:
            assert best is not None, 'no best lambda yet: run --stage select first, or pass --best-lambda / --ul-lambdas'
            lams = [best]
        return [spec(lam, 'structured', 'unlikelihood') for lam in lams]
    raise ValueError(stage)


def all_specs():
    # every run planned so far, in sheet order
    specs = stage_specs('sweep')
    if best_lambda() is not None:
        specs += stage_specs('ablation') + stage_specs('unlikelihood')
    elif args.ul_lambdas is not None:
        specs += stage_specs('unlikelihood')
    return specs


def run_paths(s, seed):
    cfg = osw.object_swap_config(s['ranking_weight'], args.margin, s['swap'], s['loss_type'],
                                 **BEST_SPEC, epochs=args.epochs, seed=seed)
    if args.tiny:
        cfg['batch_size'] = 8
        cfg['log_every'] = 5
    name = osw.run_name(cfg)
    return {
        'cfg': cfg,
        'name': name,
        'run_dir': os.path.join(RUNS_DIR, name),
        'captions': os.path.join(RESULTS_DIR, f'clipcap_{name}_val.json'),
        'eval': os.path.join(RESULTS_DIR, f'clipcap_{name}_val.eval.json'),
        'per_image': os.path.join(RESULTS_DIR, f'clipcap_{name}_val.per_image.json'),
        'swap_stats': os.path.join(RESULTS_DIR, f'clipcap_{name}_val.swap_stats.json'),
        'model': os.path.join(MODELS_DIR, f'clipcap_{name}_mapping.pt'),
    }


def run_done(p):
    needed = [p['captions'], p['model'], p['swap_stats']]
    if not (args.skip_eval or args.tiny):
        needed.append(p['eval'])
    return all(os.path.exists(f) for f in needed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def negatives_cache(split, swap):
    return None if args.tiny else os.path.join(CACHE_DIR, f'{split}_negatives_{swap}.pkl')


def load_swap_inputs():
    # object list, co-occurrence matrix, regex patterns, CHAIR ground truth
    object_data = osw.load_object_data(os.path.join(EVAL_DIR, 'coco_objects.json'))
    chair_gt = {split: osw.load_chair_gt(os.path.join(EVAL_DIR, f'chair_gt_{split}.pkl')) for split in ['train', 'val']}
    categories = object_data['categories']
    # Track 3's matrix (or --cooccurrence), else the copy rebuilt by an earlier run, else rebuild it
    cached = os.path.join(CACHE_DIR, 'cooccurrence_train.pkl')
    if os.path.exists(COOCCURRENCE):
        matrix, category_to_index = osw.load_cooccurrence_matrix(COOCCURRENCE, categories)
    elif os.path.exists(cached):
        matrix, category_to_index = osw.load_cooccurrence_matrix(cached, categories)
    else:
        print('co-occurrence matrix not found, building it from chair_gt_train.pkl')
        matrix, category_to_index = osw.build_cooccurrence_matrix(chair_gt['train'], categories)
        osw.save_cooccurrence_matrix(cached, matrix, categories)
    patterns = osw.build_replacement_patterns(object_data)
    return {'object_data': object_data, 'chair_gt': chair_gt, 'matrix': matrix,
            'category_to_index': category_to_index, 'patterns': patterns}


def load_split(split):
    data = load_features(os.path.join(FEAT_DIR, f'coco_rn50x4_{split}.pkl'))
    if args.tiny:
        data = subset_images(data, 40 if split == 'train' else 30, seed=1 if split == 'train' else 2)
    return data


def split_negatives(split, data, swap, inputs, build=True):
    cache = negatives_cache(split, swap)
    if not build and cache is not None and not os.path.exists(cache):
        raise FileNotFoundError(f'{cache} is missing: run --stage prepare first')
    return osw.build_negatives(data['captions'], inputs['chair_gt'][split], inputs['object_data'],
                               inputs['patterns'], inputs['matrix'], inputs['category_to_index'],
                               random_swap=(swap == 'random'), seed=SWAP_SEED, cache_path=cache)


def load_training_data(swaps):
    """Same train/val sets as the baseline, plus the swap datasets.

    train_sets[swap]: training ObjectSwapDataset; val_swap_sets[swap]: val pairs for swap_stats.
    """
    inputs = load_swap_inputs()
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    train_data = load_split('train')
    val_data = load_split('val')
    cache = None if args.tiny else CACHE_DIR
    train_tokens = tokenize_captions(train_data['captions'], tokenizer, cache and os.path.join(cache, 'train_tokens.pkl'))
    val_tokens = tokenize_captions(val_data['captions'], tokenizer, cache and os.path.join(cache, 'val_tokens.pkl'))
    max_len = max_seq_len(train_tokens)
    val_set = CaptionDataset(val_data, val_tokens, max_len)
    print('max caption tokens:', max_len, ' train captions:', len(train_tokens), ' val captions:', len(val_set))

    train_sets = {}
    val_swap_sets = {}
    for swap in ['structured', 'random']:
        if swap in swaps:
            negs = osw.tokenize_negatives(split_negatives('train', train_data, swap, inputs, build=False), tokenizer)
            train_sets[swap] = osw.ObjectSwapDataset(train_data, train_tokens, max_len, negs, swap)
            print(f'train {swap}: {100 * train_sets[swap].negative_rate():.1f}% of captions have a negative')
        negs = osw.tokenize_negatives(split_negatives('val', val_data, swap, inputs, build=False), tokenizer)
        val_swap_sets[swap] = osw.ObjectSwapDataset(val_data, val_tokens, max_len, negs, swap)

    return {'tokenizer': tokenizer, 'val_data': val_data, 'val_set': val_set,
            'train_sets': train_sets, 'val_swap_sets': val_swap_sets}


# ---------------------------------------------------------------------------
# Stages: check, prepare
# ---------------------------------------------------------------------------

def check_environment():
    print('python', sys.version.split()[0], '| torch', torch.__version__, '| transformers', transformers.__version__)
    print('device:', device, '|', torch.cuda.get_device_name(0) if device == 'cuda' else 'no GPU')
    java = subprocess.run(['java', '-version'], capture_output=True, text=True)
    print('java:', (java.stderr or java.stdout).strip().splitlines()[0])

    for path in [os.path.join(FEAT_DIR, 'coco_rn50x4_train.pkl'), os.path.join(FEAT_DIR, 'coco_rn50x4_val.pkl'),
                 os.path.join(DATA_DIR, 'karpathy_index.json'), os.path.join(EVAL_DIR, 'chair_gt_train.pkl'),
                 os.path.join(EVAL_DIR, 'chair_gt_val.pkl'), os.path.join(EVAL_DIR, 'synonyms.txt'),
                 os.path.join(EVAL_DIR, 'coco_objects.json'), PRETRAINED_CKPT] + \
                [os.path.join(BASELINE_RESULTS, f'clipcap_baseline_seed{s}_val.per_image.json') for s in args.seeds]:
        assert os.path.exists(path), 'missing file: ' + path
    print('data files: ok')

    # Phase 1's loss test (never run there): pretrained ClipCap on real val captions and their swaps
    inputs = load_swap_inputs()
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    val_data = load_features(os.path.join(FEAT_DIR, 'coco_rn50x4_val.pkl'))
    model = build_model(BEST_SPEC['mapping'], BEST_SPEC['prefix_length'])
    model = load_pretrained_checkpoint(model, PRETRAINED_CKPT).to(device).eval()
    smoke = osw.smoke_test_loss(model, tokenizer, val_data, inputs['chair_gt']['val'], inputs['object_data'],
                                inputs['patterns'], inputs['matrix'], inputs['category_to_index'])
    print('loss smoke test: ok')

    # the unlikelihood loss on the same kind of batch: finite, and it reaches the mapping network
    batch = []
    rng = random.Random(0)
    for c in val_data['captions']:
        entry = inputs['chair_gt']['val'].get(int(c['image_id']))
        result = None if entry is None else osw.generate_swap(
            c['caption'], osw.true_objects(entry), inputs['object_data'], inputs['patterns'],
            inputs['matrix'], inputs['category_to_index'], rng=rng)
        if result is not None:
            batch.append((tokenizer.encode(c['caption']), c['clip_embedding'], tokenizer.encode(result['caption'])))
        if len(batch) == 8:
            break
    tokens, labels, rows = osw.pack_swap_batch(batch)
    features = torch.as_tensor(val_data['features'])[rows].to(device)
    parts = osw.object_swap_loss(model, tokens.to(device), labels.to(device), features, loss_type='unlikelihood')
    parts['loss'].backward()
    assert torch.isfinite(parts['loss']), parts
    assert any(p.grad is not None for p in model.clip_project.parameters()), 'unlikelihood loss has no gradient'
    model.zero_grad(set_to_none=True)
    print(f"unlikelihood loss: ok (ranking_loss {float(parts['ranking_loss']):.3f})")
    del model

    # Track 1's self-test: human captions score > 0 and never hallucinate
    import track1_eval as t1
    rows = [r for r in t1.index if r['split'] == 'val'][:20]
    preds = {r['image_id']: r['captions'][0] for r in rows}
    gts = {r['image_id']: r['captions'] for r in rows}
    lang = t1.compute_language_metrics(gts, preds)
    assert all(lang[m] > 0 for m in ['BLEU-4', 'METEOR', 'CIDEr']), lang
    chair, _ = t1.compute_chair(preds, t1.load_gt_objects('val'))
    assert chair['n_hallucinated'] == 0
    print('evaluation (Java, METEOR, CIDEr, CHAIR): ok')

    write_json(smoke, os.path.join(RESULTS_DIR, 'loss_smoke_test.json'))
    print('ALL CHECKS PASSED')


def prepare():
    """Negatives for train and val (both swap types), the 100-negative spot checks, and a review sheet."""
    inputs = load_swap_inputs()
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    rates = {}
    for split in ['train', 'val']:
        data = load_split(split)
        if not args.tiny:
            tokenize_captions(data['captions'], tokenizer, os.path.join(CACHE_DIR, f'{split}_tokens.pkl'))
        for swap in ['structured', 'random']:
            negs = split_negatives(split, data, swap, inputs)
            rates[f'{split}_{swap}'] = sum(n is not None for n in negs) / len(negs)
            print(f'{split} {swap}: {100 * rates[f"{split}_{swap}"]:.1f}% of captions have a negative')

    # automatic spot check: 100 structured and 100 random negatives from train captions
    import track1_eval as t1
    examples = []
    for r in t1.index:
        if r['split'] == 'train' and r['image_id'] in inputs['chair_gt']['train']:
            objects = osw.true_objects(inputs['chair_gt']['train'][r['image_id']])
            examples += [(c, objects) for c in r['captions']]
    checks = {}
    for swap in ['structured', 'random']:
        print(f'{swap} swaps:')
        result = osw.spot_check_swaps(examples, inputs['object_data'], inputs['patterns'], inputs['matrix'],
                                      inputs['category_to_index'], n=100, seed=0, random_swap=(swap == 'random'))
        checks[swap] = result
        print('  errors:', result['errors'], ' checked:', result['checked'])
    write_json({'negative_rates': rates, 'spot_check': checks}, os.path.join(RESULTS_DIR, 'spot_check.json'))

    # sheet for the team's manual review: fill in is_error (1/0) and a note, then report the error rate
    rng = random.Random(1)
    pool = list(examples)
    rng.shuffle(pool)
    sheet = os.path.join(RESULTS_DIR, 'manual_review_negatives.csv')
    if os.path.exists(sheet):
        print('review sheet exists, not overwriting:', sheet)
    else:
        rows = []
        for swap in ['structured', 'random']:
            for caption, objects in pool:
                if sum(r['swap'] == swap for r in rows) >= 100:
                    break
                result = osw.generate_swap(caption, objects, inputs['object_data'], inputs['patterns'],
                                           inputs['matrix'], inputs['category_to_index'],
                                           random_swap=(swap == 'random'), rng=rng)
                if result is not None:
                    rows.append({'id': len(rows), 'swap': swap, 'original': caption, 'negative': result['caption'],
                                 'source': result['source'], 'target': result['target'], 'is_error': '', 'note': ''})
            pool = pool[len(pool) // 2:]  # different captions for the second swap type
        write_csv(rows, sheet)
        print('saved review sheet:', sheet)


# ---------------------------------------------------------------------------
# Stages: training (sweep, ablation, unlikelihood)
# ---------------------------------------------------------------------------

def score(caption_path, attempts=3):
    # Track 1's evaluate(); its Java tokenizer step can fail at random on the cluster.
    # Retry, and never let scoring stop training: the captions are saved and can be scored later.
    import track1_eval as t1
    for attempt in range(1, attempts + 1):
        try:
            t1.evaluate(caption_path, 'val', spice=False)
            return True
        except AssertionError as e:
            print(f'scoring failed (attempt {attempt} of {attempts}): {e!r}')
            time.sleep(30)
    print('WARNING: could not score', os.path.basename(caption_path), '- captions are saved, score it later')
    return False


def run_one(s, seed, data):
    """Train (or resume), decode val, score, val swap stats. Returns False if stopped at the time limit."""
    p = run_paths(s, seed)
    if run_done(p):
        print(f"{p['name']}: already done")
        return True
    if minutes_left() < 15:
        print('not enough time left in this job, stopping here')
        return False

    # leave ~25 min after training for the epoch-end val loss, captioning, scoring and swap stats
    train_deadline = None if deadline is None else deadline - 25 * 60
    print(f"{p['name']}: training ({minutes_left():.0f} min left in this job)")
    model, run_log = osw.train_object_swap_run(p['cfg'], data['train_sets'][s['swap']], data['val_set'],
                                               p['run_dir'], device, deadline=train_deadline)
    if not run_log.get('finished'):
        print(f"{p['name']}: stopped at the time limit, checkpoint saved - submit the job again")
        return False

    if not os.path.exists(p['model']):
        shutil.copy(os.path.join(p['run_dir'], 'mapping_final.pt'), p['model'])
    decode_split(model, data['tokenizer'], data['val_data'], p['captions'], device)
    if not (args.skip_eval or args.tiny):
        score(p['captions'])

    # does the trained model rank val positives above their swaps? (margin hinge for every run, so comparable)
    stats = {swap: osw.swap_stats(model, data['val_swap_sets'][swap], device, margin=args.margin)
             for swap in ['structured', 'random']}
    write_json(stats, p['swap_stats'])
    print(f"  val gap (neg - pos NLL): structured {stats['structured']['gap']:.3f}, random {stats['random']['gap']:.3f}")
    print('  train hours:', round(run_log['train_seconds'] / 3600, 2))
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return True


def train_stage(stage):
    specs = stage_specs(stage)
    jobs = [(s, seed) for s in specs for seed in args.seeds]
    mine = jobs[args.worker::args.n_workers]
    print(f'{stage}: {len(jobs)} runs, worker {args.worker} of {args.n_workers} takes {len(mine)}')
    for s, seed in mine:
        print('  ', run_paths(s, seed)['name'])
    if all(run_done(run_paths(s, seed)) for s, seed in mine):
        print('all done')
        write_runs_sheet()
        return 0

    data = load_training_data({s['swap'] for s, _ in mine})
    status = 0
    for s, seed in mine:
        if not run_one(s, seed, data):
            status = EXIT_TIME_LIMIT
            break
        write_runs_sheet()
    write_runs_sheet()
    return status


# ---------------------------------------------------------------------------
# Stages: select, summary
# ---------------------------------------------------------------------------

def spec_label(s):
    return f"lambda={s['ranking_weight']:g} swap={s['swap']} loss={s['loss_type']}"


def spec_results(s):
    # eval results for every seed of a spec, or None if any seed is not scored yet
    paths = [run_paths(s, seed)['eval'] for seed in args.seeds]
    if not all(os.path.exists(p) for p in paths):
        return None
    return [read_json(p) for p in paths]


def baseline_aggregate():
    return read_json(os.path.join(BASELINE_RESULTS, 'baseline_summary.json'))['aggregate']


def select_lambda():
    """Lowest mean val CHAIRi among lambda > 0 whose mean val CIDEr is within --max-cider-drop of the baseline."""
    import track1_eval as t1
    base = baseline_aggregate()
    floor = base['CIDEr']['mean'] - args.max_cider_drop / 100
    rows = []
    for s in stage_specs('sweep'):
        results = spec_results(s)
        if results is None:
            print(f'{spec_label(s)}: not all seeds scored yet')
            continue
        agg = t1.aggregate_seeds(results)
        rows.append({'lambda': s['ranking_weight'], 'CIDEr': agg['CIDEr']['mean'], 'CHAIRi': agg['CHAIRi']['mean'],
                     'CHAIRs': agg['CHAIRs']['mean'], 'eligible': s['ranking_weight'] > 0 and agg['CIDEr']['mean'] >= floor})

    print(f"baseline CIDEr {100 * base['CIDEr']['mean']:.2f}, floor {100 * floor:.2f} (max drop {args.max_cider_drop:g})")
    for r in rows:
        print(f"  lambda {r['lambda']:<5g} CIDEr {100 * r['CIDEr']:6.2f}  CHAIRi {100 * r['CHAIRi']:5.2f}  "
              f"CHAIRs {100 * r['CHAIRs']:5.2f}" + ('' if r['eligible'] else '  (not eligible)'))

    if len(rows) < len(stage_specs('sweep')):
        print('sweep not finished, not selecting yet')
        return 1
    eligible = [r for r in rows if r['eligible']]
    if not eligible:
        print('no lambda > 0 keeps CIDEr above the floor: decide as a team, then pass --best-lambda')
        return 1
    best = min(eligible, key=lambda r: (r['CHAIRi'], -r['CIDEr']))
    write_json({'best_lambda': best['lambda'], 'rule': 'lowest mean val CHAIRi among lambda > 0 with mean val '
                f"CIDEr >= baseline - {args.max_cider_drop:g} points", 'baseline_CIDEr': base['CIDEr']['mean'],
                'seeds': args.seeds, 'sweep': rows}, os.path.join(RESULTS_DIR, 'best_lambda.json'))
    print('best lambda:', best['lambda'], '-> results/best_lambda.json')
    return 0


def aux_trend(run_dir):
    # mid-week check: first and last training records of the logged loss parts
    records = osw.read_aux_log(os.path.join(run_dir, 'aux_loss_log.jsonl'))
    if not records:
        return None
    first, last = records[0], records[-1]
    return {'first_step': first['step'], 'last_step': last['step'],
            'ranking_loss': [first['ranking_loss'], last['ranking_loss']],
            'gap': [first['negative_nll'] - first['positive_nll'], last['negative_nll'] - last['positive_nll']],
            'pair_accuracy': [first['pair_accuracy'], last['pair_accuracy']]}


def summary():
    import track1_eval as t1
    baseline_files = {s: os.path.join(BASELINE_RESULTS, f'clipcap_baseline_seed{s}_val.per_image.json')
                      for s in args.seeds}
    out = {'margin': args.margin, 'epochs': args.epochs, 'seeds': args.seeds, 'best_lambda': best_lambda(),
           'baseline': baseline_aggregate(), 'runs': [], 'incomplete': []}

    for s in all_specs():
        results = spec_results(s)
        if results is None:
            out['incomplete'].append(spec_label(s))
            continue
        paths = {seed: run_paths(s, seed) for seed in args.seeds}
        print(f'\n=== {spec_label(s)} ===')
        rows = osw.compare_to_baseline(baseline_files, {seed: paths[seed]['per_image'] for seed in args.seeds},
                                       t1.paired_bootstrap)
        agg = t1.aggregate_seeds(results)
        for m in METRICS:
            scale = 1 if m == 'avg_len' else 100
            print(f"  {m:<11}{agg[m]['mean'] * scale:8.2f} +- {agg[m]['std'] * scale:.2f}")

        stats = {seed: read_json(paths[seed]['swap_stats']) for seed in args.seeds
                 if os.path.exists(paths[seed]['swap_stats'])}
        trends = {seed: aux_trend(paths[seed]['run_dir']) for seed in args.seeds}
        for seed in args.seeds:
            t = trends[seed]
            if t is not None:
                print(f"  seed {seed} train: ranking_loss {t['ranking_loss'][0]:.3f} -> {t['ranking_loss'][1]:.3f}, "
                      f"gap {t['gap'][0]:.3f} -> {t['gap'][1]:.3f}")
            if seed in stats:
                print(f"  seed {seed} val gap: structured {stats[seed]['structured']['gap']:.3f}, "
                      f"random {stats[seed]['random']['gap']:.3f}")
        out['runs'].append({**s, 'aggregate': agg, 'bootstrap': rows, 'swap_stats': stats, 'aux_trend': trends})

    if out['incomplete']:
        print('\nnot all seeds scored yet:', ', '.join(out['incomplete']))
    write_json(out, os.path.join(RESULTS_DIR, 'phase2_summary.json'))
    write_runs_sheet()
    print('\nsaved', os.path.join(RESULTS_DIR, 'phase2_summary.json'))
    return 0


# ---------------------------------------------------------------------------
# Run sheet (results/runs.csv): one row per planned run, rewritten by every worker
# ---------------------------------------------------------------------------

def write_csv(rows, path):
    tmp = f'{path}.{os.getpid()}.tmp'  # several workers may write at once
    with open(tmp, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ['empty'])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def run_status(p):
    if os.path.exists(p['eval']):
        return 'scored'
    if os.path.exists(p['captions']):
        return 'decoded'
    if os.path.exists(os.path.join(p['run_dir'], 'mapping_final.pt')):
        return 'trained'
    if os.path.exists(os.path.join(p['run_dir'], 'last.pt')):
        return 'in progress'
    return 'not started'


def write_runs_sheet():
    jobs = [(s, seed) for s in all_specs() for seed in args.seeds]
    rows = []
    for j, (s, seed) in enumerate(jobs):
        p = run_paths(s, seed)
        row = {'run': p['name'], 'worker': j % args.n_workers, 'lambda': s['ranking_weight'], 'swap': s['swap'],
               'loss_type': s['loss_type'], 'margin': args.margin, 'seed': seed, 'status': run_status(p),
               'train_hours': '', 'gpus': '', 'final_val_loss': ''}
        log_path = os.path.join(p['run_dir'], 'run_log.json')
        if os.path.exists(log_path):
            run_log = read_json(log_path)
            row['train_hours'] = round(run_log['train_seconds'] / 3600, 2)
            row['gpus'] = ' / '.join(run_log.get('gpus', [run_log['gpu']]))
            if run_log['epochs']:
                row['final_val_loss'] = round(run_log['epochs'][-1].get('val_loss', float('nan')), 4)
        scores = read_json(p['eval'])['scores'] if os.path.exists(p['eval']) else {}
        for m in METRICS:
            row[m] = round(scores[m] * (1 if m == 'avg_len' else 100), 2) if m in scores else ''
        stats = read_json(p['swap_stats']) if os.path.exists(p['swap_stats']) else {}
        row['val_gap_structured'] = round(stats['structured']['gap'], 4) if stats else ''
        row['val_pair_acc_structured'] = round(stats['structured']['pair_accuracy'], 4) if stats else ''
        trend = aux_trend(p['run_dir'])
        row['train_ranking_loss_first_last'] = '' if trend is None else \
            f"{trend['ranking_loss'][0]:.3f} -> {trend['ranking_loss'][1]:.3f}"
        rows.append(row)
    write_csv(rows, os.path.join(RESULTS_DIR, 'runs.csv'))


# ---------------------------------------------------------------------------

def main():
    print('stage:', args.stage, '| output:', OUT, '| time limit:',
          'none' if deadline is None else f'{args.time_limit_min:.0f} min')
    for pkg in ['punkt', 'punkt_tab', 'wordnet']:
        nltk.download(pkg, quiet=True)  # tokenizer data for CHAIR object extraction
    patch_clipcap()

    if args.stage == 'check':
        check_environment()
        return 0
    if args.stage == 'prepare':
        prepare()
        return 0
    if args.stage in ('sweep', 'ablation', 'unlikelihood'):
        return train_stage(args.stage)
    if args.stage == 'select':
        return select_lambda()
    return summary()


if __name__ == '__main__':
    sys.exit(main())
