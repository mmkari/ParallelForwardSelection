import subprocess, sys, multiprocessing
import numpy as np
from scipy import stats
import shutil, os, math, time
from GetNextCore_PFS import get_next_core
from HelperFunctions import *
from functools import partial

# calculates p-value of a feature
def calc_p_value(feat_id, split_path, target_vals, num_samples):
	x = np.loadtxt(split_path+"/feature_"+str(feat_id)+".txt", usecols=range(3,3+num_samples))
	return stats.linregress(x, target_vals)[3]	# return just the fourth value (p-value)

def get_p_values(split_path, targetsfile, n_feats, n_samples):
	# first check whether p-values have been calculated for used data
	calc_pvals = False
	if os.path.exists(split_path+"/P_VALUES/p_values.txt"):
		with open(split_path+"/P_VALUES/p_values.txt", 'r') as fin:
			line = fin.readline()
			if not targetsfile in line:
				# p-values are for some other data, need to calculate them
				calc_pvals = True
			else:
				# this is the correct file, read the remaining lines as ID - p-value pairs
				print "Found pre-calculated p-values for features. Loading them from memory...\n"
				ids = []
				p_vals = []
				for line in fin:
					parts = line.split(' ')	# split at space
					ids.append(int(parts[0]))
					p_vals.append(float(parts[1]))
				# convert to arrays
				ids = np.array(ids)
				p_vals = np.array(p_vals)
	else:
		# no p-value file was found, calculate them
		calc_pvals = True
		
	if calc_pvals == True:
		print "Calculating p-values for %d features...\n" % (int(n_feats))
		targets = np.loadtxt(targetsfile, delimiter='\n')	# loads target values into array
		
		# handle p-value calculations with a multiprocessing pool
		count_cpus = multiprocessing.cpu_count()
		pool = multiprocessing.Pool(processes=count_cpus-1)
		# create the processes
		partial_expression = partial(calc_p_value, split_path=split_path, target_vals=targets, num_samples=n_samples)
		ids = np.arange(1,int(n_feats)+1)
		r = pool.map_async(partial_expression, ids)
		pool.close()
		r.wait()
		p_vals = np.array(r.get())	# convert results to np array
		
		ids = np.argsort(p_vals)	# get sorted order
		p_vals = p_vals[ids]		# sort p_values in that order
		ids = ids+1					# make indexes into IDs
		
		# store the p-values
		if not os.path.exists(split_path+"/P_VALUES"):
			os.makedirs(split_path+"/P_VALUES")
		with open(split_path+"/P_VALUES/p_values.txt", 'w') as fout:
			fout.write(targetsfile)
			for i in range(len(ids)):
				fout.write("\n%d %f" % (ids[i], p_vals[i]))
	
	return p_vals, ids

# simple initialization that initializes k=|M| paths to core t=1
def initialize_paths_simple(M_list):
	print "Using simple initialization to initialize %d sets to one feature\n" % (len(M_list))
	# create k current sets with one distinct feature from M
	current_sets = [[item] for item in M_list]
	k = len(M_list)
	if k == 1:
		print "The number of paths is one, adding another dummy line to ensure correct I/O operations\n"
		current_sets.append('--merged: line added to ensure correct I/O')
		k = 2
	return current_sets, k

# returns a current set containing the features listed in the result for a particular path, or returns an empty list if problems
def initialize_path_to_solution_in_file(filename, path_number):
	print "Initializing one path to the solution found in file\n"
	# read the feature ids from the file, load it as a 2d list
	feature_ids = load_2d_list_from_file(filename)
	# select the indices at a given path number
	if path_number-1 >= 0 and path_number-1 < len(feature_ids):	
		feature_ids = feature_ids[path_number-1]
		# check that the solution is a list of feature IDs
		if isinstance(feature_ids, (list,)):
			return feature_ids
		else:
			return []
	else:
		return []
	# calling function should check for an empty list


def main_fun(classifier, base_path, data_path, N_feats, N_folds, L, R, k, init_type, N_iters):
	# Check that the name of the classifier is valid
	valid_classifier_names = ('LINREG', 'LASSO', 'RANFOR', 'SVM')
	if not any(name == classifier for name in valid_classifier_names):
		raise ValueError("\'%s\' is not a valid classifier name, use one of: %s\n" % (classifier, ', '.join(str(name) for name in valid_classifier_names)))
	
	# split input data by features into separate files
	pwd = os.getcwd()
	temp_path = pwd+"/TEMP"
	split_path = temp_path+"/SPLIT_DATA"
	split_geno_data(data_path+"/genotype/TRAIN.dos", split_path)
	
	# INITIALIZE CURRENT SET(S):
	# -- SIMPLE CASE (initializes sets to one of k most important features)
	if init_type == 'simple':
		# determine most promising features for use in initialization
		# these features are determined by their p-value
		with open(data_path+"/genotype/TRAIN.dos", 'r') as dataf:
			n_samples = len(dataf.readline().split(' '))-3	# determine number of training samples, needed when calculating p-value
		ids_by_importance = get_p_values(split_path, data_path+"/targets/targets_TRAIN.txt", N_feats, n_samples)[1]	# take just the ranked IDs
		
		# create k current sets of one feature by choosing features from top k p-values
		current_sets, k = initialize_paths_simple(ids_by_importance[:k])
		
	elif init_type == 'file':
		print "Loading a solution from file as the initial set of this experiment\n"
		k = max(2,k)	# there must be at least two paths for certain functions to work
		with open(pwd+"/RESULTS/best_solution_"+str(classifier)+".txt") as f:
			line = f.readline()
			current_sets = [[int(item) for item in line.split(',')]]
		for other in range(1,k):
			current_sets.append('--merged')	# add a meaningless line to prevent line from being interpreted as a list of lines in file I/O
	
	print current_sets
	
	# remove features selected in initialization
	feats_considered = []
	for path in range(k):
		if not any(word in current_sets[path] for word in ('merged','terminated')):
			feats_considered.append(np.setdiff1d(np.arange(1,int(N_feats)+1), current_sets[path]))
		else:
			feats_considered.append([])
	
	# trackers for indexing
	r_cur = 1			# current indexing level
	b_cur = r_cur + 1	# fall-back indexing level
	d_cur = 0			# indicates number of exhausted levels
	
	# initialize folders
	base_path = pwd
	res_path_cores = base_path+"/RESULTS/CORES"
	res_path_scores = base_path+"/RESULTS/SCORES"
	geno_path = temp_path+"/GENOS"
	index_path = pwd+"/TEMP/INDEXING"
	path_list = (res_path_cores, res_path_scores, geno_path, index_path)
	clean_directories(path_list)

	classifier_out_path = base_path+"/CLASSIFIER_output/"
	cvscore_path = base_path+"/CLASSIFIER_output/CV_SCORES"
	
	logs_path = temp_path+"/LOGS"
	clean_directories((logs_path,))
	
	# prepare base-set files for each path so that they contain the initialized features!
	for path in range(1,k+1):
		if isinstance(current_sets[path-1], (list,)):
			# the path has feature indexes, add them to the base file
			with open(temp_path+"/current_base_"+str(path)+".txt", 'wb') as outfile:		
				for i in current_sets[path-1]:
					with open(pwd+"/TEMP/SPLIT_DATA/feature_"+str(i)+".txt",'rb') as infile:
						shutil.copyfileobj(infile, outfile)
		else:
			# create an empty file (the file is not needed)... might not need to create it either
			open(temp_path+"/current_base_"+str(path)+".txt", 'w').close()
	
	# iterate over cores
	i_last = -9	# marks the last completed iteration, set to undefined at first
	start_iter = len(current_sets[0])+1	# the first iteration will compute core X, where X is the number of features at its end
	end_iter = start_iter+N_iters
	for i in range(start_iter,end_iter):
		
		print "starting work on Core %d\n" % (i)
		
		# at the beginning, clear previous results
		clean_directories((cvscore_path,))
		
		# determine number of considered features from first unterminated path
		all_paths_terminated = True
		for path in range(k):
			# check if this path is unterminated
			if not any(word in current_sets[path] for word in ('merged','terminated')):
				num_considered = len(feats_considered[path])	# get number of unselected features
				all_paths_terminated = False
				break
		
		if all_paths_terminated:
			# all paths have terminated, stop calculating cores
			break
		
		# check condition for falling back or initiating exhaustion
		if d_cur == 0 and num_considered < L:
			print "going to fall to a lower indexing level or start exhausing the current list at level %d\n" % r_cur
			if r_cur > b_cur:
				print "returning to indexing level "+str(b_cur)+"\n"
				# there's a level to fall back to (b_cur)
				# load indices of level b_cur
				#feats_considered = np.loadtxt(index_path+"/indices_R"+str(b_cur)+".txt", dtype=np.uint64, delimiter=',')	# commas should separate values, newlines paths
				feats_considered = load_2d_list_from_file(index_path+"/indices_R"+str(b_cur)+".txt")
				# remove indices selected since storage, do it for each path
				for path in range(k):
					if not any(word in current_sets[path] for word in ('merged','terminated')):
						feats_considered[path] = np.setdiff1d(feats_considered[path], current_sets[path])
				
				# update index trackers
				r_cur = b_cur
				b_cur += 1
				num_considered = len(feats_considered[0])
			else:
				print "starting exhaustion mode\n"
				# we have reached the end of the indexing stage
				# start exhausting the current level
				d_cur = r_cur
				
		# endif
		
		# check if the current indexing level has been exhausted
		if num_considered == 0:
			if d_cur >= 3:
				print "we have exhausted the current level of indexing, moving one down...\n"
				# move down one level
				d_cur -= 1
				# load corresponding indices
				feats_considered = load_2d_list_from_file(index_path+"/indices_R"+str(d_cur)+".txt")
				# remove indices selected since storage
				for path in range(k):
					if not any(word in current_sets[path] for word in ('merged','terminated')):
						feats_considered[path] = np.setdiff1d(feats_considered[path], current_sets[path])
				# update count of considered features
				num_considered = len(feats_considered[0])
			else:
				# we have exhausted all indexed features, exit the function
				print "we have exhausted all available features...\n"
				break
				
		# determine next core
		current_sets, feats_considered, r_cur = get_next_core(classifier, current_sets, feats_considered, i, base_path, data_path, N_folds, classifier_out_path, r_cur, b_cur, d_cur, path_list, split_path, L, R)
		
		# Check that the returned current sets do not contain duplicates
		current_sets = find_merged_paths(current_sets)
		
		# print status message
		print "Completed core "+str(i)+"...\n"
		i_last = i
	# remove left-over files
	for path in range(1,k+1):
		os.remove(temp_path+"/current_base_"+str(path)+".txt")
	shutil.rmtree(classifier_out_path+"/CV_SCORES")
	
	print "Completed Core "+str(i_last)+" before termination\n" 
		


if __name__ == "__main__":
	# parse all arguments: base_path, data_path
	classifier = sys.argv[1]
	base_path = sys.argv[2]
	data_path = sys.argv[3]
	N_feats = sys.argv[4]
	N_folds = int(sys.argv[5])
	L = int(sys.argv[6])
	R = float(sys.argv[7])
	k = int(sys.argv[8])
	N_iters = int(sys.argv[9])
	# check for additional parameter indicating type of initialization
	if len(sys.argv) > 10:
		init_type = sys.argv[10]
	else:
		init_type = 'simple'
	
	# flush all print buffers
	sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) 
	main_fun(classifier, base_path, data_path, N_feats, N_folds, L, R, k, init_type, N_iters)



