# ParallelForwardSelection
Code for running the Parallel Forward Selection process on a SLURM managed computer cluster. This code was written to perform feature selection on genetic data and to enable the operation of wrapper type selection in genome wide scale.

The execution is started with the `submit_PFS.sh` script. It calls the `ParallelForwardSelection.py` file's main function that manages the iterations in the forward-driven parallel feature selection method. Each individual iteration is then managed by the functions in the `GetNextCore_PFS.py` file. Depending on the used classifier, which is either linear or nonlinear in nature, the feature evaluation tasks are then dispathed in groups as SLURM jobs with the `submit_LinregJob.sh` and `submit_NonlinregJob.sh` scripts. These scripts evoke new jobs that operate on the `LinearRegression_Job.py` and `NonLinear_Job.py` files. Some off the common functions have been placed in the `HelperFunctions.py` file.

The Parallel Forward Selection method is a container for forward selection procedures that are operated in parallel: 

<img src=img/PFS.png width=586 height=400 />

>>**A**: Path \#2 is merged to path \#1 at the end of the second iteration as they have become identical.
**B**: Path \#1 is terminated as the validation score no longer improves.
**C**: The remaining paths are terminated as their score no longer improves or the maximum number of iterations has been reached.

The PFS method includes a speedup heuristic to reduce its compuational load and to help it scale to the genome wide scale of millions of candidate features. The speedup heuristic is illustrated in the following figure:

<img src=img/Heuristic_3.png width=586 height=464 />

>>**A**: All available features are evaluated and ranked.
**B**: The best feature is added to the set of selected features while only the top R fraction of the best performing features are considered in the next iteration.
**C**: The process of selecting the best and eliminating the worst 1-R fraction of the considered features is continued until the size of the set of considered features falls below some predefined threshold value L. At this point the index *1* stored after the first elimination is reloaded and the same process continues from there.
**D**: the number of considered features has again falled below the threshold and this time index *2* is restored from memory.
**E**: Finally all indexes have been restored once and there is no earlier index to fall back to. At this point the heuristic enters the exhaustion stage.
**F**: features are selected from the topmost current index until the selection of further features no longer improves the performance of the solution. If the topmost index is completely exhausted, the process continues from the index below that until it terminates.
