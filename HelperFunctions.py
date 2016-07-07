import subprocess, sys, multiprocessing
import numpy as np
import shutil, os
import math, time


def dispatch_feature_evaluation_jobs(classifier, feat_list, path, job_args, group_size):
	
	base_path,data_path,N_folds,result_path, cvscore_path, geno_path, split_path = job_args
	
	print "Dispatching jobs on path "+str(path)
	
	#if any(word == classifier for word in ('RANFOR', 'SVM', 'LINREG')):
	if classifier == 'RANFOR' or classifier == 'SVM' or classifier == 'LINREG' or classifier == 'LASSO':
		# dispatch evaluation jobs in collections of roughly ten features to reduce number of jobs
		num_groups = math.ceil(len(feat_list)/float(group_size))
		portion = len(feat_list)/num_groups
		print "Dispatching jobs in %d groups\n" % (int(num_groups))
		for i in range(int(num_groups)):
			
			start_ind = round(i*portion)
			end_ind = round((i+1)*portion)
			
			# submit collection as a job
			if classifier == 'RANFOR' or classifier == 'SVM':
				subprocess.Popen(['sbatch','submit_NonlinregJob.sh',str(base_path),str(data_path),str(split_path),str(N_folds),str(result_path),str(path), str(classifier)] + [str(feat_id) for feat_id in feat_list[start_ind:end_ind]])
			elif classifier == 'LINREG' or classifier == 'LASSO':
				subprocess.Popen(['sbatch','submit_LinregJob.sh',str(base_path),str(data_path),str(split_path),str(N_folds),str(result_path),str(path), str(classifier)] + [str(feat_id) for feat_id in feat_list[start_ind:end_ind]])
	else:
		# classifier not supported


def dispatch_feature_evaluation_jobs_in_batches(classifier, feat_list, num_batches, path, job_args, time_limit, max_fails, group_size):
	# dispatch jobs in batches, waiting for previous to finish or time out before dispatching the next
	# calculate estimated amount of jobs per batch
	portion = len(feat_list)/num_batches
	cur_score_path = job_args[4]+"/path_"+str(path)
	# get first batch
	print "Dispatching the first batch of %d jobs for path %d\n" % (round(portion), path)
	start_ind = 0
	end_ind = round(portion)	# exclusive
	dispatch_feature_evaluation_jobs(classifier, feat_list[start_ind:end_ind], path, job_args, group_size)
	feats_considered, num_failures = wait_for_jobs_timed(feat_list[start_ind:end_ind], cur_score_path, time_limit)
	
	# exit early if too many failures
	if num_failures > max_fails:
		return feats_considered, num_failures
	
	# process remaining batches
	for batch_index in range(1,num_batches):
		print "dispatching batch %d of path %d\n" % (batch_index+1, path)
		start_ind = round(batch_index * portion)
		end_ind = round((batch_index + 1) * portion)	# exclusive
		dispatch_feature_evaluation_jobs(classifier, feat_list[start_ind:end_ind], path, job_args, group_size)
		batch_feats_considered, batch_num_failures = wait_for_jobs_timed(feat_list[start_ind:end_ind], cur_score_path, time_limit)
		# concatenate to previous results
		feats_considered = np.concatenate((feats_considered, batch_feats_considered))
		num_failures += batch_num_failures
		
		# exit early if too many failures
		if num_failures > max_fails:
			break	# exit early, don't waste resources working on a path that will be terminated anyway
	return feats_considered, num_failures


# defines a generator that returns line from a file, withholding the first three columns
def skip_first_cols(filename, numcols=3, delimiter=None):
	# iterates the given file, yielding lines with first three cols removed
	with open(filename, 'r') as fin:
		for line in fin:
			try:
				yield line.split(delimiter, numcols)[numcols]	# return the nth substring containing the genotype values
			except IndexError:
				continue

def load_list_from_file(filename):
	with open(filename) as f:
		contents = f.readlines()
	return contents

# Splits genotype feature vectors to separate files. ID's in [1,N_feats]
def split_geno_data(filename, path):
	
	# first check that the given file has not been split already to the given split path
	split_data = False
	if os.path.exists(path+"/current_split_geno_info.txt"):
		with open(path+"/current_split_geno_info.txt", 'r') as fin:
			line = fin.readline()
		if filename not in line:
			split_data = True
	else:
		split_data = True
	
	if split_data == True:
		# first remove existing files under given path	
		if os.path.exists(path):
			shutil.rmtree(path)
		os.makedirs(path)	# make the directory
		
		# then create the new files
		id_counter = 1
		# open the geno file
		with open(filename,'r') as f:
			# process each line as a feature
			for line in f:
				# write line to new file
				with open(path+"/feature_"+str(id_counter)+".txt",'w') as newf:
					newf.write(line)
				# store feature names
				parts = line.split()
				with open(path+"/feature_indices.txt",'a') as mapf:
					mapf.write(parts[0]+'\n')
				id_counter += 1
		# finally write the info file
		with open(path+"/current_split_geno_info.txt", 'w') as fout:
			fout.write(str(filename))

def save_list_to_file(list_to_save, filename):
	with open(filename,'w') as f:
		for item in list_to_save:
			f.write("%s\n" % item)

def save_2d_list_to_file(list_to_save, filename):
	with open(filename, 'w') as f:
		#f.writelines(','.join(str(item) for item in inner_list)+"\n" for inner_list in list_to_save)
		for line in list_to_save:
			# if line is another list, save its elements as integer feature IDs
			if isinstance(line, (list, np.ndarray)):
				f.write(','.join(str(item) for item in line)+"\n")
			else:
				# the line wasn't a list, save it as a string
				f.write(str(line)+"\n")

def load_2d_list_from_file(filename):
	outer_list = []
	with open(filename, 'r') as f:
		#result = [[int(item) for item in line.split(',')] for line in f]
		lines = f.read().rstrip('\n').split('\n')	# split to lines on newline
	# if first line does not start with '--' and does not contain commas, then there is one line whose values are newline separated
	if not lines[0].startswith("--") and ',' not in lines[0]:
		# it is a 1D list
		outer_list.append([int(item) for item in lines])
	else:
		# it is a proper 2d list
		for line in lines:
			if not line.startswith("--"):
				print line
				outer_list.append([int(item) for item in line.split(',')])
			else:
				outer_list.append(str(line))
	return outer_list

def append_feature_to_base(basefile, featfile):
	f = open(featfile, 'r')
	contents = f.read()
	f.close()
	with open(basefile, 'a') as base:
		base.write(contents)

def check_jobs_ready(id_array, score_path):
	# check each result file, if any missing, return false
	ready = True
	for item in id_array:
		res_file = score_path+"/worker_"+str(item)+".txt"
		if not os.path.exists(res_file):
			ready = False
			print "File number "+str(item)+" not ready"
			break
	return ready

def check_jobs_ready_index(id_array, score_path, left_off=0):
	# check each result file, if any missing, return false
	for i in range(left_off, len(id_array)):
		res_file = score_path+"/worker_"+str(id_array[i])+".txt"
		if not os.path.exists(res_file):
			return i
	return len(id_array)

def get_inds_of_jobs_not_finished(id_array, score_path):
	jobs_unfinished = []
	for item in id_array:
		res_file = score_path+"/worker_"+str(item)+".txt"
		if not os.path.exists(res_file):
			jobs_unfinished.append(item)
	print "Following files not ready: "+str(jobs_unfinished)
	return jobs_unfinished

def get_number_of_active_paths(current_sets):
	num_active = 0
	for line in current_sets:
		if not any(word in line for word in ('merged','terminated')):
			num_active += 1
	return num_active
	
# give time limit in seconds
def wait_for_jobs_timed(id_array, score_path, time_limit):
	# get start time
	start_time = time.time()
	idx = check_jobs_ready_index(id_array, score_path)
	current_time = time.time()
	while not idx == len(id_array) and current_time < start_time + time_limit:
		print "File number "+str(id_array[idx])+" not ready"
		time.sleep(5)
		idx = check_jobs_ready_index(id_array, score_path, idx)
		current_time = time.time()	# update current time
	# at this point either jobs are ready or time-limit has been reached
	if not idx == len(id_array):
		# some jobs were left unfinished, find those indices
		inds_not_finished = get_inds_of_jobs_not_finished(id_array, score_path)
		# take set difference and return it as the new array of considered features
		return np.setdiff1d(id_array, inds_not_finished), len(inds_not_finished)
	else:
		# return the full array of evaluated, considered features
		return id_array, 0

def collect_results(id_array, score_path):
	# collect scores for specified IDs
	num_res = len(id_array)
	scores = np.zeros(num_res)
	for i in range(num_res):
		with open("%s/worker_%d.txt" % (score_path,id_array[i]), 'r') as scoref:
			scores[i] = float(scoref.read())
	return scores
	
def clean_directories(folder_list):
	# clean each folder in list
	if isinstance(folder_list, (list, tuple)):
		for folder in folder_list:
			if os.path.exists(folder):
				shutil.rmtree(folder)
			os.makedirs(folder)

def find_merged_paths(current_sets):
	for i in range(len(current_sets)-1):
		if current_sets[i] != '--merged':
			# compare set at this index to remaining indices
			for j in range(i+1, len(current_sets)):
				if current_sets[j] != '--merged' and len(np.setdiff1d(current_sets[i], current_sets[j])) == 0:
					# the sets were the same, mark set at index j 'merged'
					current_sets[j] = '--merged'
	return current_sets

