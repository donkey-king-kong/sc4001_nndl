"""Track 1 evaluation helpers for running outside Colab (TC1 cluster).

The code below the line is copied verbatim from track_1.ipynb:
"Annotations > Helper Functions" (download), "CHAIR and Object List > Helper Functions"
and "Evaluation > Helper Functions". Do not edit it here; scores must match Track 1.

Set these environment variables before importing:
  TRACK2_DATA_DIR    folder with karpathy_index.json
  TRACK2_EVAL_DIR    folder with chair_gt_val.pkl and synonyms.txt
  TRACK2_RESULTS_DIR folder where evaluate() writes .eval.json / .per_image.json
"""

import os
import json
import pickle
import shutil
import zipfile
import subprocess

import nltk
import inflect
import numpy as np
from tqdm.auto import tqdm

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

# stop Java warnings from corrupting pycocoevalcap output (tokenizer, METEOR)
os.environ['JAVA_TOOL_OPTIONS'] = '-Xlog:disable'

DATA_DIR = os.environ['TRACK2_DATA_DIR']
EVAL_DIR = os.environ['TRACK2_EVAL_DIR']
RESULTS_DIR = os.environ['TRACK2_RESULTS_DIR']

# ---------------------------------------------------------------------------
# verbatim from track_1.ipynb
# ---------------------------------------------------------------------------

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
