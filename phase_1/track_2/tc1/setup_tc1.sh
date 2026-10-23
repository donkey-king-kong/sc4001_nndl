#!/bin/bash
# One-time setup on the TC1 head node (downloads and installs only, no training).
# Run from ~/track2/code with:  bash setup_tc1.sh
set -e

cd ~/track2

# files edited on Windows can carry \r line endings, which break bash and SLURM
sed -i 's/\r$//' code/*.sh code/*.py

mkdir -p code/logs data eval out/runs out/results out/models out/cache

module load anaconda

# 1. conda environment "track2", from the course's SC4001 environment (Python 3.12, PyTorch 2.5.1)
if conda env list | grep -q '^track2 '; then
    echo "conda env track2 already exists"
else
    # the course environment (the TC1 guide calls it SC4001.yml; on the cluster it is CZ4042_v6)
    conda create -y -n track2 --clone /tc1apps/2_conda_env/CZ4042_v6 \
        || conda env create -f /tc1apps/3_conda_yml/CZ4042_v6.yml -n track2
fi
source activate track2

# 2. extra packages: Java for METEOR / the PTB tokenizer, and the Python libraries we use
#    (TC1 already has Java 11 in /usr/bin; only install one if it is missing)
if ! command -v java >/dev/null; then
    CONDA_NO_PLUGINS=true conda install -y -c conda-forge openjdk=17
fi
pip install "transformers==4.56.1" pycocoevalcap inflect nltk tqdm huggingface_hub gdown

# 3. ClipCap's model code, with its AdamW import fixed for new transformers versions
if [ ! -d CLIP_prefix_caption ]; then
    git clone https://github.com/rmokady/CLIP_prefix_caption
fi
sed -i 's/from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup/from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup\nfrom torch.optim import AdamW/' CLIP_prefix_caption/train.py

# 4. download GPT-2 and NLTK data now, in case the GPU nodes have no internet
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download('gpt2', allow_patterns=['*.json', '*.txt', 'model.safetensors'])
import nltk
for pkg in ['punkt', 'punkt_tab', 'wordnet', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    nltk.download(pkg, quiet=True)
print('GPT-2 and NLTK data downloaded')
EOF

java -version
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"
echo "Setup done. Next: sbatch job_check.sh"
