"""Track 3 helpers as an importable module for Phase 2 (B and C: integrate and run).

Copied from phase_1/track_3/track_3.py ("Track 3 helpers" section). Changes for Phase 2 are
marked 'Phase 2:'; the rest is unchanged:
- the Track 2 namespace is the track_2 module (copied next to this file), not notebook globals
- loss_type 'unlikelihood' as the optional alternative to the margin loss
- patched_baseline can log ranking_loss / positive_nll / negative_nll during training
  (the patched caption_loss only hands 'loss' to train_run, so these were never logged)
- load_cooccurrence_matrix and swap_stats (val-set ranking check after a run)
"""

import json
import os
import pickle
import random
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import inflect
import nltk
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import track_2
from track_2 import (RUN_KEYS, CaptionDataset, collate, config_name, default_config,
                     prefix_embeddings, read_json)

BASELINE = vars(track_2)  # namespace train_run looks its helpers up in

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


def load_cooccurrence_matrix(path: str | Path, categories: Sequence[str]) -> tuple[np.ndarray, dict[str, int]]:
    """Phase 2: load the matrix written by save_cooccurrence_matrix, checking the category order."""
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    if list(data['categories']) != list(categories):
        raise ValueError(f'{path}: category order differs from coco_objects.json')
    return data['matrix'], {category: i for i, category in enumerate(categories)}


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

LOSS_TYPES = ('margin', 'unlikelihood')


def swap_span_mask(tokens, labels, n):
    """Phase 2: [2B, L] mask of the negative tokens that differ from their positive (the swapped span).

    Span = negative tokens after the longest common prefix and before the longest common
    suffix of the pair. Positive rows and rows without a negative are all False.
    """
    counts = (labels != -100).sum(dim=1).tolist()
    rows = tokens.tolist()
    mask = torch.zeros(labels.shape, dtype=torch.bool)
    for i in range(n):
        b = counts[n + i]
        if b == 0:
            continue
        pos, neg = rows[i][:counts[i]], rows[n + i][:b]
        shorter = min(len(pos), b)
        k = 0
        while k < shorter and pos[k] == neg[k]:
            k += 1
        s = 0
        while s < shorter - k and pos[-1 - s] == neg[-1 - s]:
            s += 1
        mask[n + i, k:b - s] = True
    return mask.to(labels.device)


def object_swap_loss(model, tokens, labels, features, *, ranking_weight=1.0, margin=0.2, loss_type='margin'):
    """Positives and negatives in one GPT-2 forward pass, sharing one mapping-network pass.

    tokens/labels: [2B, L] from pack_swap_batch; features: [B, 640] raw CLIP.
    The GPT-2 part mirrors Track 2's caption_loss line for line, so 'nll' equals
    caption_loss on the positives (checked in smoke_test_loss).

    loss_type 'margin': the Phase 1 sequence-level hinge on per-caption mean NLL.
    loss_type 'unlikelihood' (Phase 2, optional comparison): mean of -log(1 - p(token)) over
    the swapped tokens of each negative (swap_span_mask), so only the swapped object is pushed down.
    'ranking_loss' is the auxiliary term of whichever loss_type is used.
    """
    assert loss_type in LOSS_TYPES, loss_type
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

    if loss_type == 'margin':
        hinge = F.relu(margin + positive - negative) * has_negative
        ranking = hinge.sum() / n_negatives.clamp_min(1)
    else:
        # log(1 - p) = log(-expm1(-nll)); clamped so a near-certain token stays finite
        span = swap_span_mask(tokens, labels, n).float()
        log_not_p = (-torch.expm1(-token_nll)).clamp_min(1e-6).log()
        ranking = -(log_not_p * span).sum() / span.sum().clamp_min(1)
    return {
        'loss': nll + ranking_weight * ranking,
        'nll': nll,
        'ranking_loss': ranking,
        'positive_nll': (positive * has_negative).sum() / n_negatives.clamp_min(1),
        'negative_nll': (negative * has_negative).sum() / n_negatives.clamp_min(1),
        # Phase 2: share of pairs where the negative already has the higher NLL
        'pair_accuracy': ((negative > positive).float() * has_negative).sum() / n_negatives.clamp_min(1),
        'n_negatives': n_negatives,
    }


class AuxLossLogger:
    """Phase 2: mean loss parts every log_every training steps, one JSON record per line.

    train_run only logs the total loss, so this records ranking_loss / positive_nll /
    negative_nll for the mid-week check. Records are {'step', 'n_steps', <part>: mean}.
    On resume, records past start_step (written after the last checkpoint by an earlier
    session) are dropped, so the log matches the resumed run.
    """
    KEYS = ('loss', 'nll', 'ranking_loss', 'positive_nll', 'negative_nll', 'pair_accuracy')

    def __init__(self, path, log_every, start_step=0):
        self.path = path
        self.log_every = log_every
        self.step = start_step
        records = [r for r in read_aux_log(path) if r['step'] <= start_step]
        with open(path + '.tmp', 'w') as fh:
            fh.writelines(json.dumps(r) + '\n' for r in records)
        os.replace(path + '.tmp', path)
        self._reset()

    def _reset(self):
        self.sums = {k: 0.0 for k in self.KEYS}  # GPU tensors after the first add, synced only on flush
        self.n = 0

    def add(self, parts):
        for k in self.KEYS:
            self.sums[k] = self.sums[k] + parts[k].detach()
        self.n += 1
        self.step += 1
        if self.step % self.log_every == 0:
            self.flush()

    def flush(self):
        if self.n == 0:
            return
        record = {'step': self.step, 'n_steps': self.n}
        for k in self.KEYS:
            record[k] = float(self.sums[k]) / self.n
        with open(self.path, 'a') as fh:
            fh.write(json.dumps(record) + '\n')
        self._reset()


def read_aux_log(path):
    # AuxLossLogger records, oldest first; [] if the run has none yet
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


@torch.no_grad()
def swap_stats(model, swap_set, device, *, margin=0.2, loss_type='margin', batch_size=200):
    """Phase 2: do negatives get a higher NLL than positives on a val ObjectSwapDataset?

    Means over the pairs that have a negative (batch means weighted by their pair count).
    Compare a trained run with the lambda = 0 run of the same seed: the gap
    negative_nll - positive_nll should grow with lambda.
    """
    model.eval()
    loader = DataLoader(swap_set, batch_size=batch_size, shuffle=False, collate_fn=pack_swap_batch)
    features = torch.as_tensor(swap_set.features)
    keys = ('ranking_loss', 'positive_nll', 'negative_nll', 'pair_accuracy')
    sums = {k: 0.0 for k in keys}
    total = 0.0

    for tokens, labels, rows in tqdm(loader, desc='swap stats'):
        with torch.autocast('cuda', dtype=torch.float16, enabled=device == 'cuda'):
            parts = object_swap_loss(model, tokens.to(device), labels.to(device), features[rows].to(device),
                                     margin=margin, loss_type=loss_type)
        n_pairs = float(parts['n_negatives'])
        for k in keys:
            sums[k] += float(parts[k]) * n_pairs
        total += n_pairs

    out = {k: sums[k] / max(total, 1.0) for k in keys}
    out['gap'] = out['negative_nll'] - out['positive_nll']
    out['n_pairs'] = int(total)
    return out


# ---------------------------------------------------------------------------
# Training: Track 2's train_run with the ranking term added
# ---------------------------------------------------------------------------

SWAP_RUN_KEYS = RUN_KEYS + ['ranking_weight', 'margin', 'swap', 'loss_type']


def object_swap_config(ranking_weight, margin=0.2, swap='structured', loss_type='margin', **overrides):
    # Track 2 defaults (the baseline's settings) plus the Track 3 knobs (Phase 2: loss_type)
    assert swap in ('structured', 'random'), swap
    assert loss_type in LOSS_TYPES, loss_type
    cfg = default_config(**overrides)
    cfg.update({'ranking_weight': float(ranking_weight), 'margin': float(margin), 'swap': swap,
                'loss_type': loss_type})
    return cfg


def run_name(cfg):
    # margin runs keep the Phase 1 names; the unlikelihood loss has no margin
    if cfg.get('loss_type', 'margin') == 'unlikelihood':
        return (f"objswap_{cfg['swap']}_ul_lam{cfg['ranking_weight']:g}_"
                f"{config_name(cfg)}_seed{cfg['seed']}")
    return (f"objswap_{cfg['swap']}_lam{cfg['ranking_weight']:g}_m{cfg['margin']:g}_"
            f"{config_name(cfg)}_seed{cfg['seed']}")


@contextmanager
def patched_baseline(ranking_weight, margin, ns=None, *, loss_type='margin', aux_log=None):
    """Make Track 2's train_run / dataset_loss / benchmark (in namespace ns) use the ranking term.

    - collate: ObjectSwapDataset items -> pack_swap_batch; plain (tokens, row) items
      (the val set) -> the original collate. With ranking_weight 0 the negatives
      are dropped, so batches are exactly the baseline's.
    - caption_loss: a packed [2B] batch -> object_swap_loss; anything else (val loss,
      ranking_weight 0) -> the original caption_loss, so val_loss stays comparable.
    - RUN_KEYS: resume also checks ranking_weight / margin / swap.
    - amp_dtype: fp16, the precision all baseline seeds used (as in train_baseline.py).
    - Phase 2: aux_log (an AuxLossLogger) gets the loss parts of every training step.
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
        parts = object_swap_loss(model, tokens, labels, features, ranking_weight=ranking_weight,
                                 margin=margin, loss_type=loss_type)
        if aux_log is not None and model.training:
            aux_log.add(parts)
        return parts['loss']

    caption_loss._track3 = True
    ns.update(caption_loss=caption_loss, collate=collate, RUN_KEYS=SWAP_RUN_KEYS,
              amp_dtype=lambda device: torch.float16)
    try:
        yield
    finally:
        ns.update(saved)


def checkpoint_step(run_dir, cfg, n_train):
    """Phase 2: global step saved in run_dir/last.pt, or None if there is no checkpoint."""
    path = os.path.join(run_dir, 'last.pt')
    if not os.path.exists(path):
        return None
    try:
        # mmap: only the step counters are read, not the weights and Adam moments
        state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except (RuntimeError, TypeError):  # older torch, or a non-zip checkpoint
        state = torch.load(path, map_location='cpu', weights_only=False)
    return state['epoch'] * (n_train // cfg['batch_size']) + state['step']


def train_object_swap_run(cfg, train_set, val_set, run_dir, device, aux_log=True, **kwargs):
    """Track 2's train_run with the ranking loss; same resume/deadline/max_steps behaviour.

    val_set is a plain CaptionDataset; its val_loss is the plain caption NLL.
    Phase 2: with aux_log, the loss parts go to run_dir/aux_loss_log.jsonl (AuxLossLogger).
    """
    assert isinstance(train_set, ObjectSwapDataset), 'train_set must be an ObjectSwapDataset'
    assert train_set.swap == cfg['swap'], (train_set.swap, cfg['swap'])
    # train_run returns a finished run without looking at its config, so check here
    log_path = os.path.join(run_dir, 'run_log.json')
    if os.path.exists(log_path):
        done_cfg = read_json(log_path)['config']
        for k in SWAP_RUN_KEYS:
            done = done_cfg.get(k, 'margin' if k == 'loss_type' else None)
            assert done == cfg[k], f'{run_dir} was trained with a different {k}'

    logger = None
    finished = os.path.exists(log_path) and os.path.exists(os.path.join(run_dir, 'mapping_final.pt'))
    if aux_log and cfg['ranking_weight'] != 0 and not finished:
        os.makedirs(run_dir, exist_ok=True)
        start = checkpoint_step(run_dir, cfg, len(train_set)) or 0
        logger = AuxLossLogger(os.path.join(run_dir, 'aux_loss_log.jsonl'), cfg['log_every'], start)

    with patched_baseline(cfg['ranking_weight'], cfg['margin'], loss_type=cfg['loss_type'], aux_log=logger):
        try:
            return BASELINE['train_run'](cfg, train_set, val_set, run_dir, device, **kwargs)
        finally:
            if logger is not None:
                logger.flush()


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
