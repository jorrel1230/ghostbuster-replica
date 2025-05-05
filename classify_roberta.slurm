#!/bin/bash
#SBATCH --job-name=classify-roberta-gpt_w_example
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem-per-cpu=16G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --constraint=gpu80

#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=jorrel@princeton.edu

module purge
module load anaconda3/2024.6
conda activate ghost

cd /home/jr8867/ghostbuster-replica
python ./roberta/clasify_roberta.py ./data/selected/gpt_w_example --output ./classifications/roberta/gpt_w_example.csv