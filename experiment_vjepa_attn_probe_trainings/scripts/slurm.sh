#!/bin/bash
#SBATCH --nodes=1
#SBATCH -G A100:1
#SBATCH -c 32
#SBATCH --mem=120GB
#SBATCH --time=0-24:00:00
#SBATCH -p grete:shared
#SBATCH -C ssd
#SBATCH --mail-type=BEGIN,END,FAIL          
#SBATCH --mail-user=h.siddiqui@stud.uni-goettingen.de
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# 1. Navigate to your project root directory (Slurm starts in the submission dir, but this is safe)
cd $SLURM_SUBMIT_DIR

# 2. Activate your environment if needed (uncomment if conda is required)
# source $(conda info --base)/etc/profile.d/conda.sh
conda activate vbbf

# 3. Execute your module training command
python scripts/train.py --config configs/config_03.json
