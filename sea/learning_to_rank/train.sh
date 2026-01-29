#!/bin/bash -eux
#SBATCH --job-name=sea-ltr
#SBATCH --account sci-demelo-computer-vision
#SBATCH --nodelist gx01,gx03,gx04,gx05,gx06,gx25,gx28
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu-batch
#SBATCH --cpus-per-task 16
#SBATCH --mem 60000
#SBATCH --time 24:00:00
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user vincent.eichhorn@student.hpi.uni-potsdam.de
#SBATCH --output /sc/home/vincent.eichhorn/cs-search-engine-architecture/_jobs/job_sea-ltr-%j.log


export PATH="/sc/home/vincent.eichhorn/conda3/bin:$PATH"
cd /sc/home/vincent.eichhorn/cs-search-engine-architecture
poetry install

for lr in 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6
do
  for temp in 0.05 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
  do 
    poetry run python -m sea.learning_to_rank.train --learning_rate $lr --temperature $temp
  done
done
