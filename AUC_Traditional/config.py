import os
from metrics import auc_m
seed = 1234


datas=['balance', 'dermatology', 'ecoli', 'new-thyroid', 'pageblocks', 'segmentImb', 'shuttle', 'svmguide2', 'yeast']

# models=['HingeAUCMLin', 'expAUCMLin', 'LSAUCMLin']
models = ['LSAUCMLin']

score = auc_m

penalty='l2'
C=0.5
model_name='Bsmote'
method='Bsmote'
data_name='balance'
base_path='data_index/'+data_name+'.npz/'
tot_X=base_path+'tot_X.npy'
tot_Y=base_path+'tot_Y.npy'
train_ids=base_path+'shuffle_train.npy'
test_ids=base_path+'shuffle_test.npy'
val_ids=base_path+'shuffle_val.npy'
result_data='results/'

if not os.path.exists(result_data):
    os.makedirs(result_data)

log_dir=result_data+method+'.log'
result_pkl=result_data + 'results.pkl'
result_csv=result_data + 'result_table.csv'
result_ours=result_data +'ours.csv'
model_save = result_data + 'model_'
freq_path = result_data + 'freq.csv'
param_path = result_data + 'best_param.csv'

'''
Note for the path of results:
    since we wanna save all results in one pkl and csv table, u only need to set these paths once
    and change the method (or model name) with the corresponding estimator in this file. 
    This code will process them automically. 
    In this case, the form of results are as follows:
        pkl:
            a nest dict
            {'dataset_name' : {'method': [result1, result2,.... i.e, result on each split]}...}
        csv:
            a 2-D csv table
                            method1,        method2,    ....
        dataset_name1       [mean, std]     [mean, std]
        dataset_name2       ...             ....
        
'''
