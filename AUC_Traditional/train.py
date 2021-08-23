__authors__ = "Statusrank"
__copyright__ = 'baoshilong@iie.com'

from cv import CrossValidation
from config import *
import numpy as np 
from metrics import auc_m
from utils import setup_seed
from AUVM import expAUCMLin,HingeAUCMLin,LSAUCMLin
from myLog import MyLog
import pandas as pd 
setup_seed(seed)

def train(cur_log):
    cur_log.info("model name:%s,data name:%s,method name:%s,start!" % (model_name, data_name, method))
    
    estimator = None

    # define the gridsearch parameters here.
    params = {
            'lambda1': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                        0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006], 
            'numiter': [10, 50, 80, 100], 
            'iterinner': [30],
            'thresh': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
            "stepsize": [0.1, 0.5]}
    
    # some not important params.
    lambda1 = 0.001
    numiter = 100  
    iterinner = 100
    thresh = 0.5  
    stepsize = 10
    verbose = True

    if model_name == "expAUCMLin": # Ours2
        estimator = expAUCMLin(lambda1, numiter, iterinner, thresh, stepsize, verbose)
    elif model_name == "HingeAUCMLin": # Ours3  
        estimator = HingeAUCMLin(lambda1, numiter, iterinner, thresh, stepsize, verbose, _cython=True)
    elif model_name == "LSAUCMLin": # Ours1
        estimator = LSAUCMLin(lambda1, numiter, iterinner, thresh, stepsize, verbose) 
    else:
        raise ValueError

    # see cv.py for more details
    cv = CrossValidation(
        cur_log=cur_log,
        dataname=data_name,
        method=method,
        estimator=estimator,
        scorer = auc_m,
        n_jobs = 5, # change the n_jobs, if u wanna accelerate your program, more details see cv.py
        data_path={'base_path': base_path, 
                   'tot_X': tot_X, 
                   'tot_Y':tot_Y,
                   'train_ids': np.load(train_ids), 
                   'val_ids': np.load(val_ids), 
                   'test_ids': np.load(test_ids),
                   'pkl': result_pkl,
                   'csv': result_csv,
                   'results':result_ours,
                   'model_save': model_save,
                   'freq_path': freq_path
                   },
        param_grid = params,
        whether_record=False
    )
    cv.fit()
    
def tot_train(load_best_param = False):
    for cur_model_name in models:
        for cur_data_name in datas:
            set_config(cur_data_name,cur_model_name)
            cur_log=MyLog(log_dir)
            train(cur_log)

def set_config(cur_data_name,cur_model_name):
    global model_name, method, data_name, base_path, tot_X, tot_Y, train_ids, \
           test_ids, val_ids, log_dir, result_data, result_pkl, result_csv, result_ours
    
    model_name = cur_model_name 
    method = cur_model_name
    data_name = cur_data_name

    base_path = 'data_index/' + data_name + '.npz/'
    tot_X = base_path + 'tot_X.npy'
    tot_Y = base_path + 'tot_Y.npy'
    train_ids = base_path + 'shuffle_train.npy'
    test_ids = base_path + 'shuffle_test.npy'
    val_ids = base_path + 'shuffle_val.npy'

    result_data = 'results/'

    if not os.path.exists(result_data):
        os.makedirs(result_data)
    log_dir = result_data + method + '.log'
    result_pkl = result_data + 'results.pkl'
    result_csv = result_data + 'result_table.csv'
    result_ours = result_data + method +  '.csv'
    model_save = result_data + 'model_'
    freq_path = result_data + 'freq.csv'
if __name__=="__main__":
    
    tot_train()
