#!/bin/bash
#SBATCH --job-name=hifdata                ## Job name
#SBATCH --account=indikar1            ## PI's account
#SBATCH --partition=standard,largemem         ## Partition/queue name
#SBATCH --nodes=1                    ## Number of nodes
#SBATCH --ntasks=1                   ## Number of tasks
#SBATCH --cpus-per-task=1            ## Cores per task
#SBATCH --time=24:00:00               ## Max runtime
#SBATCH --mem=100g                   ## Memory
#SBATCH --mail-user=jpic@umich.edu  ## Email notifications
#SBATCH --output=/home/jpic/logs/%x-%A_%a.out      ## Output file: jobname-jobid_arraytaskid.out
#SBATCH --array=0-255                 ## Array job: adjust range based on number of datasets

# Run Python script with current array task ID
python dataset_statistics_job.py $SLURM_ARRAY_TASK_ID
