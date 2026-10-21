"""Track 2 baseline on the NTU CCDS TC1 cluster (or any Linux machine with an NVIDIA GPU).

Trains the chosen baseline (transformer mapping, prefix 40, chosen by the sweep) for each seed,
one after another. Each seed resumes from its checkpoint (including checkpoints copied from Colab),
then decodes val, evaluates with Track 1's code, and once all 3 seeds are done writes the summary.

Usage (inside a SLURM job, see job_baseline.sh):
    python train_baseline.py --check                 # environment check, quick
    python train_baseline.py --time-limit-min 340    # train/resume all seeds

Exit codes: 0 = everything requested is done, 3 = stopped at the time limit (submit again), other = error.
"""

import argparse
import os
import subprocess
import sys
import time

START = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--root', default=os.path.expanduser('~/track2'), help='folder with data/, eval/, out/, CLIP_prefix_caption/')
parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--time-limit-min', type=float, default=None, help='save and stop before this many minutes (job limit)')
parser.add_argument('--check', action='store_true', help='only check the environment and files')
parser.add_argument('--device', default='cuda')
parser.add_argument('--skip-eval', action='store_true', help='testing only: skip scoring')
parser.add_argument('--tiny', action='store_true', help='testing only: tiny data, small batches')
args = parser.parse_args()

# chosen by the mini sweep (results/sweep_summary.json): transformer p40, val CIDEr 103.9
BEST_SPEC = {'mapping': 'transformer', 'prefix_length': 40}
ALL_SEEDS = [0, 1, 2]
EXIT_TIME_LIMIT = 3

ROOT = args.root
DATA_DIR = os.path.join(ROOT, 'data')
EVAL_DIR = os.path.join(ROOT, 'eval')
OUT = os.path.join(ROOT, 'out')
RESULTS_DIR = os.path.join(OUT, 'results')
RUNS_DIR = os.path.join(OUT, 'runs')
MODELS_DIR = os.path.join(OUT, 'models')
CACHE_DIR = os.path.join(OUT, 'cache')
for d in [RESULTS_DIR, RUNS_DIR, MODELS_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

os.environ['TRACK2_DATA_DIR'] = DATA_DIR
os.environ['TRACK2_EVAL_DIR'] = EVAL_DIR
os.environ['TRACK2_RESULTS_DIR'] = RESULTS_DIR
os.environ['JAVA_TOOL_OPTIONS'] = '-Xlog:disable'  # Java warnings corrupt pycocoevalcap output

# ClipCap's model code, and track_2.py (same folder, or the parent folder in the repo)
sys.path.insert(0, os.path.join(ROOT, 'CLIP_prefix_caption'))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import GPT2TokenizerFast  # noqa: E402

transformers.AdamW = torch.optim.AdamW  # older transformers; setup_tc1.sh also patches train.py

import track_2  # noqa: E402
from track_2 import (CaptionDataset, build_model, config_name, decode_split,  # noqa: E402
                     default_config, load_features, max_seq_len, read_json, subset_images,
                     tokenize_captions, train_run, write_json)

# The runs started on Colab T4s in fp16 mixed precision. Always use fp16, so a resumed run on a
# newer GPU (which would otherwise pick bf16) keeps the same precision and all seeds match.
track_2.amp_dtype = lambda device: torch.float16
amp_dtype = track_2.amp_dtype

device = args.device
deadline = None if args.time_limit_min is None else START + args.time_limit_min * 60


def minutes_left():
    return float('inf') if deadline is None else (deadline - time.time()) / 60


def seed_paths(seed):
    cfg = default_config(**BEST_SPEC, epochs=args.epochs, seed=seed)
    if args.tiny:
        cfg['batch_size'] = 8
        cfg['log_every'] = 5
    name = f'baseline_{config_name(cfg)}_seed{seed}'
    return {
        'cfg': cfg,
        'run_dir': os.path.join(RUNS_DIR, name),
        'captions': os.path.join(RESULTS_DIR, f'clipcap_baseline_seed{seed}_val.json'),
        'eval': os.path.join(RESULTS_DIR, f'clipcap_baseline_seed{seed}_val.eval.json'),
        'model': os.path.join(MODELS_DIR, f'clipcap_baseline_seed{seed}_mapping.pt'),
    }


def describe_checkpoints():
    for seed in ALL_SEEDS:
        p = seed_paths(seed)
        last = os.path.join(p['run_dir'], 'last.pt')
        if os.path.exists(os.path.join(p['run_dir'], 'mapping_final.pt')):
            print(f'  seed {seed}: training finished')
        elif os.path.exists(last):
            state = torch.load(last, map_location='cpu', weights_only=False)
            print(f"  seed {seed}: checkpoint at epoch {state['epoch']} step {state['step']} "
                  f"(trained {state['run_log']['train_seconds'] / 3600:.1f} h so far)")
        else:
            print(f'  seed {seed}: no checkpoint, will start from scratch')


def check_environment():
    print('python', sys.version.split()[0], '| torch', torch.__version__, '| transformers', transformers.__version__)
    assert torch.cuda.is_available(), 'no GPU visible - run this inside a SLURM job with --gres=gpu:1'
    print('GPU:', torch.cuda.get_device_name(0), '| mixed precision:', amp_dtype('cuda'))
    java = subprocess.run(['java', '-version'], capture_output=True, text=True)
    print('java:', (java.stderr or java.stdout).strip().splitlines()[0])

    for path in [os.path.join(DATA_DIR, 'coco_rn50x4_train.pkl'), os.path.join(DATA_DIR, 'coco_rn50x4_val.pkl'),
                 os.path.join(DATA_DIR, 'karpathy_index.json'), os.path.join(EVAL_DIR, 'chair_gt_val.pkl'),
                 os.path.join(EVAL_DIR, 'synonyms.txt')]:
        assert os.path.exists(path), 'missing file: ' + path
    print('data files: ok')

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    model = build_model(BEST_SPEC['mapping'], BEST_SPEC['prefix_length']).to('cuda')
    print('GPT-2 loaded offline: ok,', sum(p.numel() for p in model.clip_project.parameters()) // 10**6, 'M mapping params')
    del model, tokenizer

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

    describe_checkpoints()
    print('ALL CHECKS PASSED')


def write_summary():
    import track1_eval as t1
    import numpy as np

    seed_results = []
    timing = []
    curve_lines = ['seed,epoch,train_loss,val_loss']
    for seed in ALL_SEEDS:
        p = seed_paths(seed)
        seed_results.append(read_json(p['eval']))
        run_log = read_json(os.path.join(p['run_dir'], 'run_log.json'))
        timing.append({'seed': seed, 'gpu': run_log['gpu'], 'train_hours': run_log['train_seconds'] / 3600,
                       'median_sec_per_step': float(np.median([e['session_sec_per_step'] for e in run_log['epochs']])),
                       'val_loss_per_epoch': [e['val_loss'] for e in run_log['epochs']]})
        for e in run_log['epochs']:
            curve_lines.append(f"{seed},{e['epoch']},{e['train_loss']:.5f},{e['val_loss']:.5f}")

    with open(os.path.join(RESULTS_DIR, 'baseline_loss_curves.csv'), 'w') as fh:
        fh.write('\n'.join(curve_lines) + '\n')

    agg = t1.aggregate_seeds(seed_results)
    ref_path = os.path.join(DATA_DIR, 'clipcap_pretrained_seed0_val.eval.json')
    reference = read_json(ref_path)['scores'] if os.path.exists(ref_path) else {}

    print('\nBaseline', BEST_SPEC, f'- {args.epochs} epochs, val, greedy, {len(ALL_SEEDS)} seeds')
    print(f"{'metric':<12}{'mean':>10}{'std':>10}{'pretrained':>12}")
    for m in t1.METRIC_ORDER:
        if m not in agg:
            continue
        scale = 1 if m == 'avg_len' else 100
        ref = reference.get(m)
        ref_text = f'{ref * scale:>12.2f}' if ref is not None else f"{'-':>12}"
        print(f"{m:<12}{agg[m]['mean'] * scale:>10.2f}{agg[m]['std'] * scale:>10.2f}{ref_text}")
    print()
    for t in timing:
        print(f"seed {t['seed']}: {t['train_hours']:.2f} h training in total, last GPU {t['gpu']}")

    write_json({'config': BEST_SPEC, 'epochs': args.epochs, 'seeds': ALL_SEEDS, 'aggregate': agg,
                'per_seed': [r['scores'] for r in seed_results], 'timing': timing,
                'note': 'training started on Colab T4 and resumed on TC1 V100; both use fp16 mixed precision'},
               os.path.join(RESULTS_DIR, 'baseline_summary.json'))
    write_json({'config': BEST_SPEC, 'epochs': args.epochs, 'seeds': ALL_SEEDS,
                'train_settings': default_config(**BEST_SPEC, epochs=args.epochs)},
               os.path.join(RESULTS_DIR, 'baseline_plan.json'))
    print('saved baseline_summary.json, baseline_plan.json and baseline_loss_curves.csv in', RESULTS_DIR)
    if 'CIDEr' in agg and agg['CIDEr']['mean'] < 0.9:
        print('WARNING: CIDEr well below the pretrained reference - check for a bug')


def main():
    if args.check:
        check_environment()
        return 0

    print('time limit:', 'none' if deadline is None else f'{args.time_limit_min:.0f} min')
    describe_checkpoints()

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    train_data = load_features(os.path.join(DATA_DIR, 'coco_rn50x4_train.pkl'))
    val_data = load_features(os.path.join(DATA_DIR, 'coco_rn50x4_val.pkl'))
    cache = CACHE_DIR
    if args.tiny:
        train_data = subset_images(train_data, 40, seed=1)
        val_data = subset_images(val_data, 30, seed=2)
        cache = None
    train_tokens = tokenize_captions(train_data['captions'], tokenizer,
                                     cache and os.path.join(cache, 'train_tokens.pkl'))
    val_tokens = tokenize_captions(val_data['captions'], tokenizer, cache and os.path.join(cache, 'val_tokens.pkl'))
    max_len = max_seq_len(train_tokens)
    train_set = CaptionDataset(train_data, train_tokens, max_len)
    val_set = CaptionDataset(val_data, val_tokens, max_len)
    print('train captions:', len(train_set), ' val captions:', len(val_set))

    for seed in args.seeds:
        p = seed_paths(seed)
        done = os.path.exists(p['captions']) and (args.skip_eval or os.path.exists(p['eval']))
        if done:
            print(f'seed {seed}: already done')
            continue

        if minutes_left() < 15:
            print('not enough time left in this job, stopping here')
            return EXIT_TIME_LIMIT

        # leave ~25 min after training for the epoch-end val loss, captioning and scoring
        train_deadline = None if deadline is None else deadline - 25 * 60
        print(f'seed {seed}: training ({minutes_left():.0f} min left in this job)')
        model, run_log = train_run(p['cfg'], train_set, val_set, p['run_dir'], device, deadline=train_deadline)
        if not run_log.get('finished'):
            print(f'seed {seed}: stopped at the time limit, checkpoint saved - submit the job again')
            return EXIT_TIME_LIMIT

        if not os.path.exists(p['model']):
            import shutil
            shutil.copy(os.path.join(p['run_dir'], 'mapping_final.pt'), p['model'])
        decode_split(model, tokenizer, val_data, p['captions'], device)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

        if not args.skip_eval:
            import track1_eval as t1
            t1.evaluate(p['captions'], 'val', spice=False)
        print(f'seed {seed}: done')

    if args.skip_eval:
        return 0
    missing = [s for s in ALL_SEEDS if not os.path.exists(seed_paths(s)['eval'])]
    if missing:
        print('seeds still to run:', missing)
    else:
        write_summary()
    return 0


if __name__ == '__main__':
    sys.exit(main())
