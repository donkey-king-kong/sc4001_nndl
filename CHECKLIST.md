# Project Checklist

Status as of 26 Oct, based on what is in the repo (Phase 2 code in `phase_2/track_1_and_3/` is not committed yet). Legend: `[x]` done, `[~]` partly done or differs from the plan, `[ ]` not started.

Track 1 = `phase_1/track_1` (Person A), Track 2 = `phase_1/track_2` (Person B), Track 3 = `phase_1/track_3` (Person C). Phase 2 (B and C) = `phase_2/track_1_and_3`.

**Nothing in `phase_2/track_1_and_3` has been run yet**; the files only pass `python -m py_compile`.

---

## Phase 1: Baseline and components (19–25 Oct)

### Person A: data and evaluation
- [x] Extract CLIP features for all images once and save them
  - Uses **RN50x4** (640-d), not ViT-B/32. RN50x4 is the CLIP model behind ClipCap's COCO transformer checkpoint, so this is the right choice; fix the wording in the plan and report. Saved as `features/coco_rn50x4_{train,val,test}.pkl` (Karpathy split: 113,287 / 5,000 / 5,000).
- [x] Set up pycocoevalcap with Java (OpenJDK 21; BLEU-4, METEOR and CIDEr pass the self-test)
- [x] CHAIR implementation
  - Our own implementation built on LisaAnne's `synonyms.txt`, with 15/15 object-extraction unit tests passing. Ground-truth caches: `eval/chair_gt_{train,val,test}.pkl`.
- [x] Sanity check: CHAIR on ground-truth captions
  - A caption scored against its own ground truth gives CHAIRi = CHAIRs = 0.
  - Leave-one-out human captions give CHAIRi 2.56% and CHAIRs 3.63%, against 7.6% and 11.2% for the model. That is low, not zero, and the remainder is real annotation disagreement. Explain this in the report.
- [x] Extras: `paired_bootstrap`, `aggregate_seeds`, the pretrained ClipCap checkpoint evaluated on val (CIDEr 104.1, CHAIRi 7.63%)

### Person B: baseline
- [x] ClipCap running on current library versions (patches the `AdamW` import)
- [x] Baseline trained: GPT-2 frozen, transformer mapping network
- [~] Mini D1 sweep (3 epochs, seed 0, selected by val CIDEr), results in `results/sweep_summary.json`
  - [x] transformer, prefix 40: CIDEr 103.9, CHAIRi 8.11%
  - [x] transformer, prefix 10: CIDEr 101.5, CHAIRi 7.90%
  - [x] MLP, prefix 10: CIDEr 90.1, CHAIRi 7.50%
  - [ ] MLP, prefix 40: deliberately skipped (about 480M parameters, too slow). The reason is written down in the notebook; it also needs to go in the report.
  - Chosen: **transformer, prefix 40**
- [x] Chosen baseline trained with 3 seeds (10 epochs), results in `results/baseline_summary.json`
  - CIDEr 104.43 ± 0.58 · BLEU-4 31.14 ± 0.21 · METEOR 26.47 ± 0.07
  - CHAIRi 7.21 ± 0.19 · CHAIRs 10.89 ± 0.25 · length 9.79 · object recall 55.1%
- [~] Numbers in the same range as the ClipCap paper
  - The numbers are there (trained 104.4 vs pretrained checkpoint 104.1, both with greedy decoding), but nobody has written the comparison against the paper's table. The paper reports roughly CIDEr 113 with beam search, so the expected gap comes from greedy vs beam decoding. Check the paper's figures and write this up.

### Person C: negative captions
- [x] Object-word list from CHAIR's synonym list (Track 1 `coco_objects.json`, 80 categories, 403 words; reused by Track 3)
- [x] Object co-occurrence matrix, saved as `track_3/eval/cooccurrence_train.pkl`
  - Small deviation: it is built from the CHAIR ground truth (segmentation objects plus objects mentioned in the captions), not from the instance annotations alone. Mention this in the Method section.
- [x] Swap function (`generate_swap`)
  - [x] Picks one object mention (random order, retries other mentions if one fails)
  - [x] Replaces it with a frequently co-occurring object not annotated in the image (random pick from the top 10 by co-occurrence)
  - [x] Handles "a"/"an", capitalisation and plurals
  - [x] Re-runs CHAIR extraction to confirm the new object is detected (catches cases like "teddy" + "bear")
  - [~] Single GPT-2 token rule: no explicit rule. Multi-token words work anyway because the loss is computed over the whole caption, not at one token. Record this as the team's decision.
- [x] Random-swap version (`random_swap=True`)
- [~] Check about 100 negatives and record the error rate
  - `spot_check_swaps` checks 100 structured and 100 random negatives automatically, but the notebook has no saved output, so no error rate is recorded.
  - Phase 2: `run_phase2.py --stage prepare` runs the spot check and saves the counts to `results/spot_check.json`. Not run yet.
  - [ ] Manual review by the team, with the error rate recorded for the report
    - `--stage prepare` also writes `results/manual_review_negatives.csv` (100 structured + 100 random negatives with blank `is_error` / `note` columns) to fill in.
- [~] Loss implemented and tested on a toy batch
  - **The design differs from the plan.** It is a sequence-level hinge loss, `max(0, 0.2 + nll(pos) − nll(neg))`, on the per-token mean NLL of each caption. It does not compare log p(true word) with log p(swapped word) at the swap position. Either update the plan and report or change the code.
  - `smoke_test_loss` checks: NLL equals the baseline caption loss, λ = 0 equals the baseline exactly, positive NLL < negative NLL, gradients reach only the mapping network, and a batch with no negatives stays finite. **The notebook has no saved outputs, so there is no evidence it has run.**
  - Phase 2: `run_phase2.py --stage check` runs `smoke_test_loss` on the pretrained checkpoint and saves the values to `results/loss_smoke_test.json`. Not run yet.

### End-of-week milestone
- [x] Baseline metrics on the val split
- [ ] Negatives file for the training set: code is written, but no file exists yet. Phase 2 builds it with `--stage prepare` into `phase_2/track_1_and_3/cache/{train,val}_negatives_{structured,random}.pkl` (no longer `track_3/cache/`)
- [ ] Loss passing a unit test: test is written but has not been run (`--stage check` runs it)

---

## Phase 2: Main experiments 

### B and C: integrate and run
Folder `phase_2/track_1_and_3/`:
- `track_2.py`: copy of `phase_1/track_2/track_2.py`, unchanged
- `track1_eval.py`: copy of `phase_1/track_2/tc1/track1_eval.py` (Track 1 scoring, CHAIR, paired bootstrap), unchanged
- `object_swap.py`: Track 3 helpers from `phase_1/track_3/track_3.py` as a module; Phase 2 changes are marked `Phase 2:`
- `run_phase2.py`: the runner, one stage at a time: `check`, `prepare`, `sweep`, `select`, `ablation`, `unlikelihood`, `summary` (usage in its docstring). Same resume / time-limit / exit code 3 behaviour as Track 2's `train_baseline.py`, fp16 like the baseline, val only.

Track 3 input: only `phase_1/track_3/eval/cooccurrence_train.pkl`, and it is optional (rebuilt from `chair_gt_train.pkl` into `cache/` if missing, and reused from there afterwards).

- [x] Auxiliary loss added to training: total = caption NLL + λ × ranking loss (`patched_baseline`, `train_object_swap_run`; reuses Track 2's `train_run`, including resume and checkpoints)
- [~] λ sweep {0, 0.1, 0.3, 1, 3} × 3 seeds
  - [x] All five λ values set up (`--stage sweep`, change with `--lambdas`). λ = 0 is the sanity run that should reproduce the baseline.
  - [ ] Run them (15 runs, each about 1.5–2× a baseline run)
  - [~] Choose the best λ on val: `--stage select` picks the lowest mean val CHAIRi among λ > 0 whose mean val CIDEr is within 2 points of the baseline, and writes `results/best_lambda.json`. **This rule is a placeholder; agree on it as a team** (`--max-cider-drop`, or override with `--best-lambda`).
- [~] Ablation: co-occurrence vs random negatives at the best λ
  - [x] Set up at the λ chosen by `select` (`--stage ablation`), no longer fixed at λ = 1
  - [ ] Run it (3 runs)
- [~] Optional: margin loss vs unlikelihood loss
  - [x] Unlikelihood loss added (`loss_type='unlikelihood'`): −log(1 − p) over the swapped tokens of each negative only. This is closer to the plan's per-word comparison than the margin loss. Margin stays the default.
  - [ ] Run it (`--stage unlikelihood`, at the best λ or `--ul-lambdas`; 3 runs)
- [~] Split runs across GPUs and log every run to the spreadsheet
  - [x] `--worker i --n-workers k` gives run j of a stage to worker j % k, so one process per GPU
  - [x] `results/runs.csv` is the run sheet: one row per planned run with worker, status, train hours, GPUs, val metrics and the ranking checks. Every worker rewrites it.
  - [ ] SLURM job script for Phase 2 (none yet; copy the `phase_1/track_2/tc1/job_baseline.sh` pattern)
- [x] Extra: per-seed paired bootstrap against the baseline (`compare_to_baseline`); `--stage summary` writes `results/phase2_summary.json`

### Person A: analysis while runs finish
- [~] Tune and select only on val, never test: the code only touches val so far. Keep to this.
- [ ] CHAIR-vs-CIDEr trade-off plot
- [ ] Hallucination rate broken down by object frequency
- [ ] Hallucination rate broken down by co-occurrence strength (the `per_image.json` files plus `cooccurrence_train.pkl` have the data needed)
- [~] Caption length and object recall: `evaluate` already computes `avg_len` and `obj_recall` for every run; they still need to go into a results table

### Mid-week check 
- [ ] Confirm the auxiliary loss is going down
  - Training now logs `ranking_loss`, `positive_nll`, `negative_nll` and `pair_accuracy` every `log_every` steps to `runs/<run>/aux_loss_log.jsonl` (trimmed back to the checkpoint on resume). `--stage summary` and `runs.csv` show the first → last values.
- [ ] Confirm p(swapped word) / negative NLL is going up relative to the positive
  - Gap fixed in code: besides the training log, every finished run gets `results/clipcap_<run>_val.swap_stats.json` (val positive vs negative NLL gap and pair accuracy, for both swap types). Compare each run with the λ = 0 run of the same seed; the gap should grow with λ.
- [ ] If there is no effect, debug before running more

### code freeze
- [ ] No new experiments after this date

---

## Phase 3: Final evaluation and report (2–7 Nov)

### Final evaluation (first day, once only)
- [ ] Run the chosen configs on the Karpathy test split (test features and `chair_gt_test.pkl` are ready)
- [ ] Report mean ± std over 3 seeds

### Report (10 pages max, A4, Arial 10pt, single column)
- [ ] Introduction and review of existing techniques: ClipCap, CHAIR, POPE, contrastive and negative training (B)
- [ ] Method: baseline, negative generation, loss (C)
- [ ] Experimental setup, results, plots, tables (A)
- [ ] Discussion, limitations, failure cases (all, C leads)
- [ ] AI Use Statement, acknowledging existing code and pretrained models (B)
- [ ] About 6 failure-case images: where the method helped, where it didn't, and new errors it introduced
- [ ] Limitations: CHAIR only covers COCO's 80 categories; metrics are noisy; compute was limited; some negatives are wrong (with the measured rate)

### Code ZIP (A)
- [ ] README: dependencies, data preparation, feature extraction, and the exact commands that reproduce the main tables (the current `README.md` is only a title)
- [~] Comments in the key files (Tracks 1–3 already have docstrings and section notes; check them before submission)

### Last 2 days
- [ ] Everyone proofreads the whole report
- [ ] Run-through: each person explains a part they didn't write
