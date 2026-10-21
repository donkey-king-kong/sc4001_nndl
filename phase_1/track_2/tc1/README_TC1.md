# Track 2 baseline on the CCDS TC1 GPU cluster

Finishes the 3 baseline seeds (transformer, prefix 40, 10 epochs) on TC1's V100 GPUs,
continuing from the checkpoints saved on Colab. One GPU per user, so the seeds run one after another:
seed 0, then 1, then 2. Each job runs up to 6 hours, saves a checkpoint and submits the next job by itself.

## 1. Files to download

**From the GitHub repo folder (already on your laptop):**

| File | Put it in (on TC1) |
|---|---|
| `phase_1/track_2/track_2.py` | `~/track2/code/` |
| `phase_1/track_2/tc1/train_baseline.py`, `track1_eval.py`, `job_check.sh`, `job_baseline.sh`, `setup_tc1.sh` | `~/track2/code/` |
| `phase_1/track_1/features/coco_rn50x4_val.pkl` | `~/track2/data/` |
| `phase_1/track_1/data/karpathy_index.json` | `~/track2/data/` |
| `phase_1/track_1/results/clipcap_pretrained_seed0_val.eval.json` | `~/track2/data/` |
| `phase_1/track_1/eval/chair_gt_val.pkl`, `phase_1/track_1/eval/synonyms.txt` | `~/track2/eval/` |

**From Google Drive** (first close all Colab tabs so nothing is still writing checkpoints):

| File or folder | Where on Drive | Put it in (on TC1) |
|---|---|---|
| `coco_rn50x4_train.pkl` | `sc4001_nndl_grp_project/phase_1/track_1/features/` | `~/track2/data/` |
| `baseline_transformer_p40_seed0/`, `..._seed1/`, `..._seed2/` (whole folders, each has `last.pt`) | `sc4001_track2/runs/` | `~/track2/out/runs/` |
| `train_tokens.pkl`, `val_tokens.pkl` (optional, saves ~2 min per job) | `sc4001_track2/cache/` | `~/track2/out/cache/` |

Final layout on TC1:

```
~/track2/
  code/   track_2.py  train_baseline.py  track1_eval.py  job_check.sh  job_baseline.sh  setup_tc1.sh
  data/   coco_rn50x4_train.pkl  coco_rn50x4_val.pkl  karpathy_index.json  clipcap_pretrained_seed0_val.eval.json
  eval/   chair_gt_val.pkl  synonyms.txt
  out/runs/   baseline_transformer_p40_seed0/last.pt  ..._seed1/last.pt  ..._seed2/last.pt
  out/cache/  train_tokens.pkl  val_tokens.pkl   (optional)
```

## 2. Connect

1. Be on NTU Wi-Fi (NTUSECURE), or connect the NTU VPN first (vpngate.ntu.edu.sg).
2. Install **PuTTY** (terminal) and **WinSCP** (file transfer).
3. **PuTTY:** Host = the TC1 head node address from the TC1 user guide, port `22`, Open. Log in with your NTU username (lower case) and password.
4. Create the folders:
   ```
   mkdir -p ~/track2/code ~/track2/data ~/track2/eval ~/track2/out/runs ~/track2/out/cache
   ```
5. **WinSCP:** protocol SFTP, same host as PuTTY, port `22`, same login. Drag the files into the folders above.

## 3. One-time setup (~20-30 min)

In PuTTY:
```
cd ~/track2/code
bash setup_tc1.sh
```
It creates the `track2` conda environment from the course's SC4001 environment, installs Java and the
missing Python packages, downloads ClipCap's code, GPT-2 and NLTK data. It ends with `Setup done`.

## 4. Check everything (~5 min)

```
cd ~/track2/code
sbatch job_check.sh
squeue -u $USER                      # wait until the job disappears from the list
cat logs/output_t2check_*.out
```
It must end with `ALL CHECKS PASSED` and list each seed's checkpoint (epoch and step).
If not, look at `cat logs/error_t2check_*.err`.

## 5. Run

```
cd ~/track2/code
sbatch job_baseline.sh
```
You can now close PuTTY and your laptop. The job continues on the cluster and resubmits itself every ~6 hours until all 3 seeds are done.

**Check progress** (log back in any time):
```
squeue -u $USER                                   # R = running, PD = waiting for a GPU
ls -t ~/track2/code/logs/output_t2base_*.out | head -1 | xargs tail -n 20
grep -h "epoch\|seed" ~/track2/code/logs/output_t2base_*.out
```

When everything is finished, the last log ends with the results table (mean ± std over 3 seeds next to the pretrained ClipCap).

## 6. Bring the results back

With WinSCP, download `~/track2/out/results/` and `~/track2/out/models/`, then upload their files to
`sc4001_nndl_grp_project/phase_1/track_2/results/` and `.../models/` on Drive.
Key files: `clipcap_baseline_seed{0,1,2}_val.json`, `baseline_summary.json`, `baseline_loss_curves.csv`,
`clipcap_baseline_seed{n}_mapping.pt`.

## Troubleshooting

| Problem | Fix |
|---|---|
| Job stays `PD` | Waiting for a free GPU; it starts by itself. The reason is shown in `squeue` |
| Log says `Could not resubmit automatically` | `cd ~/track2/code && sbatch job_baseline.sh` |
| Error mentioning `\r` or `$'\r'` | `sed -i 's/\r$//' ~/track2/code/*.sh ~/track2/code/*.py` |
| Need to stop | `scancel <jobid>` (job id from `squeue`); it resumes from the last checkpoint next time |

Rules from the TC1 guide: never run training on the head node (always `sbatch`), use only your own home
folder, and don't post the TC1 guide publicly.
