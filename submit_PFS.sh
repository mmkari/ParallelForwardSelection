#!/bin/bash
#SBATCH -J MAIN_PROGRAM
#SBATCH -o out/output.out
#SBATCH -e out/error.err
#SBATCH -p batch
#SBATCH -t 3:00:00
#SBATCH --mem=6000

#module load python/2.7.2
#module load anaconda2

# srun python ParallelForwardSelection.py <classifier> <base_path> <data_path> <num_features> <CV_folds> <L> <R> <num_paths> <num_iterations> (<filename>)
# classifier: options are LINREG, LASSO, SVM, RANFOR
# base_path: path to directory with program files
# data_path: path to directory with data files
# filename (optional): when defined, uses given file to initialize one path
srun python ParallelForwardSelection.py LINREG /path/to/base/folder /path/to/data 12000 6 100 0.4 8 50
#srun python ParallelForwardSelection.py LINREG /path/to/base/folder /path/to/data 12000 6 100 0.4 8 50 /path/to/previous/solution