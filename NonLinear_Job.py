import sys, shutil, os, subprocess, math
import numpy as np
from sklearn import linear_model, cross_validation, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from HelperFunctions import skip_first_cols

# define own scorer for cross validation: calculates correlation coefficient for each folds predicted and true labels
def corrcoef_score_func(ground_truth, predictions):
	cor_score = np.corrcoef(ground_truth, predictions)[0,1]
	# if score is NAN, return 0, else return the score
	if math.isnan(cor_score):
		return 0
	else:
		return cor_score


# copies files to nodes
def run_nonlin_groupjob_locally(feat_ids, base_path, data_path, split_path, N_folds, result_path, path_number, local_path, classifier):
	
	path_number = int(path_number)
	
	# determine current base file of this path
	if path_number == 0:
		cur_base_file = base_path+"/TEMP/current_base.txt"
	else:
		cur_base_file = base_path+"/TEMP/current_base_"+str(path_number)+".txt"	
	
	# DO NOT CREATE A FILE, READ THE CONTENTS DIRECTLY
	# read common base file
	BASE_X = np.loadtxt(skip_first_cols(cur_base_file))
	if len(BASE_X.shape) == 1:
		BASE_X = BASE_X.reshape((1,len(BASE_X)))
	# read target values
	y = np.loadtxt(data_path+"/targets/targets_TRAIN.txt")
	
	# evaluate features
	for feature_id in feat_ids:
		
		# read current candidate feature
		cur_file = split_path+"/feature_"+str(feature_id)+".txt"
		line = np.loadtxt(skip_first_cols(cur_file))
		cur_feat = line.reshape((1,len(line)))
		
		# catenate base X's to this feature, then transpose -> (samples, features)
		if BASE_X.size == 0:
			# base file was empty, simply add the current feature
			X = cur_feat
		else:
			X = np.concatenate((BASE_X, cur_feat), axis=0)
		# transpose
		X = np.transpose(X)
		
		if classifier == 'RANFOR':
			crm = RandomForestRegressor(n_estimators=100, oob_score=True)
			crm.fit(X, y)
			average = crm.oob_score_
		if classifier == 'SVM':
			# cross validate to select hyperparameters: favor smaller C for fewer support vectors and simpler models; gamma determines radius of support vectors, larger = smaller radius of influence
			candidate_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3,0.01,0.1,1],'C': [0.01,1,100]}] # C:[0.01,1,100], g:[1e-3,0.01,0.1,1]
			model = GridSearchCV(svm.SVR(), candidate_parameters, cv=3)
			average = np.average(cross_validation.cross_val_score(model, X, y, cv=int(N_folds)))	# returns average score
		
		# store Cross Validation score to file
		#result_path+"/CV_SCORES/path_"+str(path_number)+"/worker_%d.txt" % int(feature_id)
		cvscore_path = "%s/CV_SCORES/path_%s/worker_%d.txt" % (result_path, str(path_number), int(feature_id))		# important file, gathered in caller function
		with open(cvscore_path,'w') as scorefile:
			scorefile.write("%s" % average)

# copies files to nodes, INCOMPLETE: eg does no GRID search for svm hyperparameters
def run_nonlin_job_locally(num, base_path, data_path, split_path, N_folds, result_path, path_number, local_path, classifier):
	
	num = int(num)
	path_number = int(path_number)
	
	# determine current base file of this path
	if path_number == 0:
		cur_base_file = base_path+"/TEMP/current_base.txt"
	else:
		cur_base_file = base_path+"/TEMP/current_base_"+str(path_number)+".txt"	
	
	# generate a temporary genotype file containing the required features
	filenames = [cur_base_file, split_path+"/feature_"+str(num)+".txt"]
	genopath = local_path+"/TEMP/GENOS/path_"+str(path_number)	# on the local node
	genofile = genopath +"/genofile_"+str(num)+".txt"	# stored to local node
	# create path if does not exist
	if not os.path.exists(genopath):
		os.makedirs(genopath)
	with open(genofile, 'wb') as outfile:
		for f in filenames:
			with open(f,'rb') as infile:
				shutil.copyfileobj(infile, outfile)
	cvscore_path = result_path+"/CV_SCORES/path_"+str(path_number)+"/worker_%d.txt" % num		# important file, gathered in caller function
	
	# read the X and y values
	X = np.transpose(np.loadtxt(skip_first_cols(genofile, 3)))	# reads the genofile, skipping first three columns
	y = np.loadtxt(data_path+"/targets/targets_TRAIN.txt")
	
	# prepare scoring strategy: score by correlation with true labels
	if classifier == 'RANFOR':
		rfc = RandomForestRegressor(n_estimators=100)
		fold_scores = cross_validation.cross_val_score(rfc, X, y, cv=int(N_folds))
	if classifier == 'SVM':
		svrm = svm.SVR()
		fold_scores = cross_validation.cross_val_score(svrm, X, y, cv=int(N_folds))
	
	# calculate mean correlation score (of folds)
	average = np.average(fold_scores)
	
	# store Cross Validation score to file
	with open(cvscore_path,'w') as scorefile:
		scorefile.write("%s" % average)
	
	# in the end, remove unused files: old genofile
	os.remove(genofile)


# handles parameters
if __name__ == "__main__":
	base_path = sys.argv[1]
	data_path = sys.argv[2]
	split_path = sys.argv[3]
	N_folds = sys.argv[4]
	result_path = sys.argv[5]
	path_number = sys.argv[6]
	local_path = sys.argv[7]
	classifier = sys.argv[8]
	feat_id = sys.argv[9:]
	
	if len(feat_id) > 1:
		# there are multiple features to evaluate in this call, evaluate the group of features
		run_nonlin_groupjob_locally(feat_id, base_path, data_path, split_path, N_folds, result_path, path_number, local_path, classifier)	# run optimized version of multiple evaluations per job
	else:
		# there was only one feature to evaluate
		run_nonlin_job_locally(feat_id, base_path, data_path, split_path, N_folds, result_path, path_number, local_path, classifier)
	
