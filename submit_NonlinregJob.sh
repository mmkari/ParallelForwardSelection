#!/bin/bash
#SBATCH -J NonlinearJob
#SBATCH -o batchout/out.out
#SBATCH -e batchout/out.err
#SBATCH -p batch
#SBATCH -t 25:00
#SBATCH --mem-per-cpu=100

# this script needs the feature id as parameter $1
if [ "$#" -lt 7 ]; then
    echo "The following parameters are required:
	(1) path to base folder (home of TMP, data and other folders)
	(2) path to data folder
        (3) path to split data
	(4) number of cross validation folds
	(5) path to result folder
        (6) number of path (default zero for serial)
        (7-) feature IDs"
    exit
fi

# pass all arguments to Gemma_Job.py, inserting node root path as 6th parameter
srun python NonLinear_Job.py "${@:1:6}" ${TMPDIR} "${@:7}"
