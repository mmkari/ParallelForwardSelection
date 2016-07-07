import subprocess, sys, multiprocessing
import numpy as np
import shutil, os
import math, time
from HelperFunctions import *
		

def get_next_core(classifier, current_sets, feats_considered, iteration, base_path, data_path, N_folds, result_path, r_cur, b_cur, d_cur, path_list, split_path, L, R):
	
	# NEED TO BE DEFINED
	MAX_NUM_OF_FAILURES_PER_PATH = 10
	MAX_NUM_OF_JOBS_TO_DISPATCH = 4000
	MAX_TIME_PER_BATCH = 40*60	# seconds

	cvscore_path = result_path+"/CV_SCORES"
	res_path_cores, res_path_scores, geno_path, index_path = path_list	# get paths from tuple
	temp_path = base_path+"/TEMP"
	
	# Logs file
	log_file = temp_path+"/LOGS/log.txt"
	
	# following list is passed to feature evaluation dispatching function
	job_argument_list = [base_path,data_path,N_folds,result_path, cvscore_path, geno_path, split_path]
	
	# number of considered features 
	num_paths = len(feats_considered)
	num_active_paths = get_number_of_active_paths(current_sets)
	
	#num_considered = len(feats_considered[0])
	# get number of considered features from first unterminated path
	for path in range(num_paths):
		# check if this path is unterminated
		if not any(word in current_sets[path] for word in ('merged','terminated')):
			num_considered = len(feats_considered[path])	# get number of unselected features
			break
	
	# determine number of tasks per group --- affects number of jobs (each group evaluation = 1 job)
	if classifier == 'RANFOR':
		TASKS_PER_GROUP = 600.0
	elif classifier == 'SVM':
		TASKS_PER_GROUP = 30.0
		if num_active_paths * num_considered / TASKS_PER_GROUP < 100:
			# ensure the calculations are parallelized across workers
			TASKS_PER_GROUP = round(TASKS_PER_GROUP / 2)
	elif classifier == 'LINREG':
		TASKS_PER_GROUP = 200.0
	elif classifier == 'LASSO':
		TASKS_PER_GROUP = 250.0
	else:
		TASKS_PER_GROUP = 1.0
	
	# evaluate number of upcoming jobs (accounting for terminated and merged paths)
	can_dispatch_at_once = num_considered / TASKS_PER_GROUP * num_active_paths <= MAX_NUM_OF_JOBS_TO_DISPATCH
	
	# determine whether indices need to be stored this iteration and reserve memory accordingly
	if d_cur == 0 and r_cur == b_cur:
		# indices need to be stored, allocate a 2d list
		indices = ['--none' for n in range(num_paths)]	# at the end one additional feature has been selected		
	
	scores = -9 * np.ones(num_paths)	# initialize scores to unset value -9
	
	# IF THERE ARE NOT TOO MANY JOBS TO DISPATCH AT ONCE TO SLURM, loop over all paths, dispatching jobs
	if can_dispatch_at_once:
		# we may dispatch all jobs at once without SLURM complaining and dismissing any of them
		for path in range(1,num_paths+1):
			
			if not any(word in current_sets[path-1] for word in ('merged','terminated')):
				# at the beginning, clear previous results
				cur_score_path = cvscore_path+"/path_"+str(path)
				cur_geno_path = geno_path+"/path_"+str(path)
				clean_directories((cur_score_path, cur_geno_path))
				
				dispatch_feature_evaluation_jobs(classifier, feats_considered[path-1], path, job_argument_list, TASKS_PER_GROUP)
			else:
				print "Skipping merged path number %d" % path
	
	# Read previous iteration's scores to see which paths are actually still increasing (we will terminate those that do not)
	previous_score_file = "%s/score_%d.txt" % (res_path_scores, iteration-1)
	if os.path.exists(previous_score_file):
		# scores exists, load them so we can use them in later comparisons
		with open(previous_score_file, 'r') as f:
			previous_scores = [float(item) for item in f]
	else:
		# score file didn't exist, this should be the case in the first performed iteration, initialize to -5
		previous_scores = [-5.0 for path in range(num_paths)]
	
	# Now all jobs have been dispatched, start waiting for results and process them path by path
	for path in range(1,num_paths+1):
		
		if not any(word in current_sets[path-1] for word in ('merged','terminated')):
			
			# determine current path's score folder
			cur_score_path = cvscore_path+"/path_"+str(path)
			
			# wait until all jobs are ready, use time limit to ignore processes that do not complete. Count these to
			# make sure there are at most a few and if several evaluations time out, mark the path --terminated due to too many failed evaluations
			# wait for results and leave some unfinished features out if necessary
			if not can_dispatch_at_once:
				# the jobs have not yet been dispatched, dispatch those of this current path
				# at the beginning, clear previous results
				cur_geno_path = geno_path+"/path_"+str(path)
				clean_directories((cur_score_path, cur_geno_path))
				
				# check whether all jobs of this path can be dispatched at once
				if len(feats_considered[path-1]) <= MAX_NUM_OF_JOBS_TO_DISPATCH:
					# we can dispatch this path's jobs at once
					dispatch_feature_evaluation_jobs(classifier, feats_considered[path-1], path, job_argument_list, TASKS_PER_GROUP)
					feats_considered[path-1], num_failures = wait_for_jobs_timed(feats_considered[path-1], cur_score_path, MAX_TIME_PER_BATCH)
				else:
					# we must dispatch the jobs in batches
					feats_considered[path-1], num_failures = dispatch_feature_evaluation_jobs_in_batches(classifier, feats_considered[path-1], int(math.ceil(float(len(feats_considered[path-1]))/MAX_NUM_OF_JOBS_TO_DISPATCH)), path, job_argument_list, MAX_TIME_PER_BATCH, MAX_NUM_OF_FAILURES_PER_PATH, TASKS_PER_GROUP)
			else:
				# the jobs were dispatched in the previous loop for all paths, wait for results here
				feats_considered[path-1], num_failures = wait_for_jobs_timed(feats_considered[path-1], cur_score_path, MAX_TIME_PER_BATCH)	# time limit in s, Consider dividing MAX_TIME by number of paths
			
			# recompute number of considered features after potential failed jobs
			cur_num_considered = len(feats_considered[path-1])
			
			# if there were failures log them
			if num_failures > 0:
				with open(log_file, 'a') as logf:
					logf.write('There were %d failures in iteration %d for path %d --- Limit is %d.\n' % (num_failures, iteration, path, MAX_NUM_OF_FAILURES_PER_PATH))
			
			# Important, if several feature evaluations fail, the results of this path become unreliable and it needs to be marked as --terminated
			if num_failures > MAX_NUM_OF_FAILURES_PER_PATH:
				current_sets[path-1] = '--terminated: too many failed jobs (%d)' % num_failures
				continue	# to next path
			
			# Check that there are enough features to index, given that we are in indexing phase
			if d_cur == 0 and cur_num_considered < 1 + math.floor(R*(num_considered-1)):
				current_sets[path-1] = '--terminated: not enough features to proceed to next level (had %d, needed %d)' % (len(feats_considered[path-1]), 1 + math.floor(R*(num_considered-1)))
				continue
			
			# collect the results
			results = collect_results(feats_considered[path-1], cur_score_path)
			
			# give -1 score to features with NAN score so that they are not selected as the feature with the highest score
			results[np.isnan(results)] = -1
			
			
			# determine best expansion and update indices if required
			if d_cur == 0:
				# we're in indexing phase, sort all
				sorted_inds = np.argsort(results)[::-1]
				
				# if the best score is no better than the score in previous iteration, terminate this path
				if results[sorted_inds[0]] <= previous_scores[path-1]:
					# terminate this path
					current_sets[path-1] = '--terminated: score started to decrease (change was %f)' % (results[sorted_inds[0]] - previous_scores[path-1],)
					continue	# to next path
				
				# add best feature to current set
				id_best = feats_considered[path-1][sorted_inds[0]]
				current_sets[path-1].append(id_best)
				# save current set and score
				##save_list_to_file(current_set, res_path_cores+"/core_"+str(iteration)+".txt")
				append_feature_to_base(temp_path+"/current_base_"+str(path)+".txt", split_path+"/feature_"+str(id_best)+".txt")
				##save_list_to_file((results[sorted_inds[0]],), res_path_scores+"/score_"+str(iteration)+".txt")
				scores[path-1] = results[sorted_inds[0]]
				# check whether it's time to store the indices
				if r_cur == b_cur:
					# store indices before they are reduced
					#save_list_to_file(feats_considered[sorted_inds[1:]], index_path+"/indices_R"+str(b_cur)+".txt")
					# Add indices to array
					indices[path-1] = feats_considered[path-1][sorted_inds[1:]]
				# reduce the number of considered features
				count_next_considered = math.floor(R*(num_considered-1))
				feats_considered[path-1] = feats_considered[path-1][sorted_inds[1:count_next_considered+1]]
				# update the indexing level
				#r_cur += 1	# NOT HERE, at the end
			else:
				# we are in exhaustion stage, finding best expansion is enough
				ind = np.argmax(results)
				
				# if the best score is no better than the score in previous iteration, terminate this path
				if results[ind] <= previous_scores[path-1]:
					# terminate this path
					current_sets[path-1] = '--terminated: score started to decrease (change was %f)' % (results[ind] - previous_scores[path-1],)
					continue	# to next path
				
				# update current set
				id_best = feats_considered[path-1][ind]
				current_sets[path-1].append(id_best)
				# save core and score
				##save_list_to_file(current_set, res_path_cores+"/core_"+str(iteration)+".txt")
				append_feature_to_base(temp_path+"/current_base_"+str(path)+".txt", split_path+"/feature_"+str(id_best)+".txt")
				##save_list_to_file((results[ind],), res_path_scores+"/score_"+str(iteration)+".txt")
				scores[path-1] = results[ind]
				# remove added feature from list of considered features
				feats_considered[path-1] = feats_considered[path-1][np.arange(cur_num_considered)!=ind]
			# clear all temporary files 
	
	# save current sets and scores
	save_2d_list_to_file(current_sets, res_path_cores+"/core_"+str(iteration)+".txt")
	np.savetxt(res_path_scores+"/score_"+str(iteration)+".txt", scores, delimiter='\n')
	
	if d_cur == 0:
		if r_cur == b_cur:
			# indices need to be saved to file
			#np.savetxt(index_path+"/indices_R"+str(b_cur)+".txt", indices, delimiter=',')
			save_2d_list_to_file(indices, index_path+"/indices_R"+str(b_cur)+".txt")
		r_cur += 1
	
	return current_sets, feats_considered, r_cur
		


