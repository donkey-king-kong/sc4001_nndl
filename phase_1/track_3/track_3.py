"""Track 3 object swaps and ClipCap loss helpers.

The functions in this module are deliberately independent of Colab.  The
notebook can load the returned objects from Track 1's Drive artifacts and
provide the already-loaded ClipCap model for the loss smoke test.
"""

from __future__ import annotations

import json
import pickle
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_object_data(path: str | Path) -> dict:
    """Load and validate ``coco_objects.json`` from Track 1."""
    with open(path) as handle:
        data = json.load(handle)

    required = {"categories", "synonyms", "word_to_node", "double_words"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"object data is missing keys: {sorted(missing)}")
    return data


def load_chair_gt(path: str | Path) -> dict:
    """Load Track 1's ``chair_gt_<split>.pkl`` cache."""
    with open(path, "rb") as handle:
        return pickle.load(handle)


def true_objects(cache_entry: Mapping) -> set[str]:
    """Return segmentation objects unioned with every human-caption object."""
    objects = set(cache_entry.get("seg", []))
    for caption_objects in cache_entry.get("cap", []):
        objects.update(caption_objects)
    return objects


def build_cooccurrence_matrix(
    chair_gt: Mapping[int, Mapping],
    categories: Sequence[str],
) -> tuple[np.ndarray, dict[str, int]]:
    """Count image-level object co-occurrences, including the diagonal.

    ``matrix[i, j]`` is the number of images whose CHAIR object union contains
    both categories.  Unknown cache labels are ignored so a partially updated
    artifact cannot silently change matrix dimensions.
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
    with open(target, "wb") as handle:
        pickle.dump({"categories": list(categories), "matrix": matrix}, handle)


def _replacement_patterns(object_data: Mapping) -> list[tuple[re.Pattern, str]]:
    word_to_node = object_data["word_to_node"]
    phrases = sorted(word_to_node, key=lambda phrase: (len(phrase.split()), len(phrase)), reverse=True)
    return [
        (re.compile(r"(?<![A-Za-z])" + re.escape(phrase) + r"(?![A-Za-z])", re.IGNORECASE), category)
        for phrase, category in ((phrase, word_to_node[phrase]) for phrase in phrases)
    ]


def _object_mentions(caption: str, object_data: Mapping) -> list[tuple[str, str, int, int]]:
    """Find non-overlapping synonym spans as ``(text, category, start, end)``."""
    occupied: list[tuple[int, int]] = []
    mentions = []
    for pattern, category in _replacement_patterns(object_data):
        for match in pattern.finditer(caption):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            occupied.append(span)
            mentions.append((match.group(0), category, span[0], span[1]))
    return sorted(mentions, key=lambda mention: mention[2])


def _candidate_categories(
    source: str,
    image_objects: set[str],
    categories: Sequence[str],
    matrix: np.ndarray | None,
    category_to_index: Mapping[str, int] | None,
    random_swap: bool,
    rng: random.Random,
) -> list[str]:
    candidates = [category for category in categories if category != source and category not in image_objects]
    if random_swap or matrix is None or category_to_index is None or source not in category_to_index:
        rng.shuffle(candidates)
        return candidates

    source_row = matrix[category_to_index[source]]
    return sorted(candidates, key=lambda category: (source_row[category_to_index[category]], category))


def generate_swap(
    caption: str,
    image_objects: set[str],
    object_data: Mapping,
    matrix: np.ndarray | None = None,
    category_to_index: Mapping[str, int] | None = None,
    *,
    random_swap: bool = False,
    rng: random.Random | None = None,
) -> dict | None:
    """Generate one negative by replacing an object with an absent category."""
    rng = rng or random.Random()
    mentions = _object_mentions(caption, object_data)
    if not mentions:
        return None

    source_text, source, start, end = rng.choice(mentions)
    candidates = _candidate_categories(
        source,
        image_objects,
        object_data["categories"],
        matrix,
        category_to_index,
        random_swap,
        rng,
    )
    if not candidates:
        return None

    target = candidates[0]
    return {
        "caption": caption[:start] + target + caption[end:],
        "source": source,
        "target": target,
        "source_text": source_text,
        "random_swap": random_swap,
    }


def spot_check_swaps(
    examples: Sequence[tuple[str, set[str]]],
    object_data: Mapping,
    matrix: np.ndarray,
    category_to_index: Mapping[str, int],
    *,
    n: int = 100,
    seed: int = 0,
    random_swap: bool = False,
) -> dict:
    """Generate and validate up to 100 structured or random negatives."""
    rng = random.Random(seed)
    selected = list(examples)
    rng.shuffle(selected)
    selected = selected[:n]
    checked = 0
    failures = []

    for caption, image_objects in selected:
        result = generate_swap(
            caption,
            image_objects,
            object_data,
            matrix,
            category_to_index,
            random_swap=random_swap,
            rng=rng,
        )
        if result is None:
            continue
        checked += 1
        if result["target"] in image_objects or result["target"] == result["source"]:
            failures.append(result)

    return {"requested": len(selected), "checked": checked, "failures": failures}


def caption_nll(model, tokenizer, prefix, caption: str, prefix_length: int = 40):
    """Teacher-forced GPT-2 NLL for one caption and one raw CLIP feature."""
    if prefix.ndim == 1:
        prefix = prefix.unsqueeze(0)
    prefix = prefix.float()
    prefix = prefix / prefix.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    token_ids = tokenizer.encode(caption, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    if not token_ids:
        raise ValueError('caption produced no tokens')

    tokens = torch.tensor(token_ids, device=prefix.device, dtype=torch.long).unsqueeze(0)
    prefix_embeds = model.clip_project(prefix).reshape(1, prefix_length, -1)
    token_embeds = model.gpt.transformer.wte(tokens)
    inputs_embeds = torch.cat((prefix_embeds, token_embeds), dim=1)
    labels = torch.cat(
        (
            torch.full((1, prefix_length), -100, device=tokens.device, dtype=torch.long),
            tokens,
        ),
        dim=1,
    )
    return model.gpt(inputs_embeds=inputs_embeds, labels=labels).loss


def object_swap_loss(
    model,
    tokenizer,
    prefixes,
    captions: Sequence[str],
    negative_captions: Sequence[str],
    *,
    prefix_length: int = 40,
    margin: float = 0.2,
    ranking_weight: float = 1.0,
):
    """Return positive NLL, negative NLL, ranking term, and total loss."""
    if len(captions) != len(negative_captions) or len(captions) != len(prefixes):
        raise ValueError("prefixes, captions, and negative_captions must have equal length")

    positive = torch.stack([
        caption_nll(model, tokenizer, prefixes[i], captions[i], prefix_length)
        for i in range(len(captions))
    ])
    negative = torch.stack([
        caption_nll(model, tokenizer, prefixes[i], negative_captions[i], prefix_length)
        for i in range(len(captions))
    ])
    ranking = F.relu(margin + positive - negative).mean()
    return {
        "positive_nll": positive.mean(),
        "negative_nll": negative.mean(),
        "ranking_loss": ranking,
        "loss": positive.mean() + ranking_weight * ranking,
    }


def smoke_test_loss(model, tokenizer, prefixes) -> dict[str, float]:
    """Run the loss on two captions using a loaded pretrained ClipCap model."""
    captions = ["a dog beside a person", "a bird on a tree"]
    negatives = ["a zebra beside a person", "a train on a tree"]
    values = object_swap_loss(model, tokenizer, prefixes[:2], captions, negatives)
    return {name: float(value.detach().cpu()) for name, value in values.items()}