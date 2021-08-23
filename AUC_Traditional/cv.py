
__authors__ = "Statusrank"
__copyright__ = 'baoshilong@iie.com'


import pandas as pd 
import numpy as np 
import pickle as pk
from sklearn.preprocessing import Normalizer
from collections.abc import Mapping, Iterable
from numpy.ma import MaskedArray
from collections import defaultdict 
from scipy.stats import rankdata
import os, operator
from joblib import Parallel, delayed 
from metrics import * # import our score function to help select the estimator.
from Copy import clone, cross_validate 
from itertools import product
from functools import reduce, partial

class ParamGrid:
    '''
    this class is to iterate over parameter value combinations
    
    Parm:
    ------------
    param_grid: dict of str to sequence
    The parameter grid to explore, as a dictionary mapping estimator parameters to sequences of allowed values.
    
    An empty dict signifies default parameters.
    
    A sequence of dicts signifies a sequence of grids to search, and is useful to avoid exploring parameter combinations that make no sense.

    '''

    def __init__(self, param_grid):
        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError('Parameter grid is not a dict or '
                            'a list ({!r})'.format(param_grid))
        
        if isinstance(param_grid, Mapping):
            param_grid = [param_grid]
        
        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError('Parameter grid is not a '
                                'dict ({!r})'.format(grid))
            for key in grid:
                if not isinstance(grid[key], Iterable):
                    raise  TypeError('Parameter grid value is not iterable '
                                    '(key={!r}, value={!r})'
                                    .format(key, grid[key]))      
        self.param_grid = param_grid
    
    def __iter__(self):
        '''
        Iterate over the params in the grid

        Return:
            params, yield dictionaries mapping for estimator param to 
            one of its values.
        '''
        for p in self.param_grid:
            # always sort the keys of a dictionary for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params
    def __len__(self):
        '''
        Number of points on the grid
        '''
        product = partial(reduce, operator.mul)
        return sum(product(len(v) for v in p.values()) if p else 1
                    for p in self.param_grid)
    
    def __getitem__(self, ind):
        '''
        Get the params that would be 'ind-th' in iteration

        Param:
            ind: the index of iteration
        
        Return:
            params: dict of str to any
            Equal to list(self)[ind]
        '''
        # This is used to make discrete sampling without replacement memory
        # efficient.
        for sub_grid in self.param_grid:
            # XXX: could memoize information used here
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue

            # Reverse so most frequent cycling parameter comes first
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.product(sizes)

            if ind >= total:
                # Try the next grid
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                return out

        raise IndexError('ParameterGrid index out of range')



# This class is inspried by sklearn. We extended it to apply to our settings. 

class CrossValidation:
    '''
    This class is designed for only one method on one data, if u wanna run it for all methods on all datasets, 
    u should use: 
        for method, dataname in ...., and set the data_path _dict

    It reproduces CV as GridSearchCV using cross_validation from slearn, and u can use self.cv_results_ to 
    check all processes during CV.

    Attention:
        In this class, i do not record the best estimator of best params, but just record the best_params, and 
        corresponding scores for all splits on val data and test data. 

    u should use:
        CV = CrossValidation(...)
        CV.fit()
    to run the Cross validation, and it will record the results automatically (make sure u set the correct params in data_path).
    '''
    def __init__(self,
                 cur_log,
                dataname, 
                method, 
                estimator, 
                scorer,
                data_path = None, 
                param_grid = [], 
                refit = True, 
                n_jobs = None, 
                verbose = 1,
                n_splits = 15,
                return_train_score = True,
                whether_record = False):
        '''param

        dataname:  str
            the name of data set
        
        method: str
            the name of method to run

        estimator: estimator
            model which is conducted grid search 

        data_path: dict
            dict to load ur data, e.g. = {'base_path':'data...','train_features': 'data/yzy/...', 'train_label': 'data/blabla..',.... test_'label': '...'}
            .npy form

            It must include: 
                'base_path': 'the base data path'
                'tot_X': the total data X
                'tot_Y': the total data Y
                'train_ids': list[str], ids for train/val/test on all split, correspond one-to-one
                'val_ids':  list[str] 
                'test_ids': list[str] 

                'pkl': str, path to save the whole results. // for 'pkl' and 'csv' and 'model_save', i will add default values for them, so u can ignore them. default = results.pkl
                'csv': str, path to save the whole results tables. // default = table.csv
                'results': str, path to save the all intermediate results for our method, if self.whether_record = True, else cannot set. // default = ours.csv
                'model_save':
                'freq_path': 

        param_grid: dict
            dict of all hyper-prarm to exhaustively generate grid search
            
        scorer: evalution  function, must be defined !!!
            scoring function to estimate ur model 

        n_jobs: int or None, optional (default=None)
            Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context. 
            if u wanna accelerate by parallel, n_job should > 1, maybe 2, 3 or 4, -1 means using all processors.

        verbose: Int, default = 1
            Controls the verbosity: the higher, the more messages.

        n_splits: int, default = 15
            the number of splits during CV
            this param is also used to assert whether the parallels are conducted !!!
            The parallels are defaulted as multiprocessing !!!

        return_train_score: default = False
            whether return train score on each split
        
        whether_record: bool, default = False
            if True, record all results during the grid search
            if False (default), no record.

        '''
        self.cur_log=cur_log

        self.dataname = dataname
        self.method = method

        self.estimator = estimator
        self.param_grid = param_grid
        
        self.data_path = defaultdict()
        
        self.data_path['pkl'] = 'results.pkl'
        self.data_path['csv'] = 'table.csv'
        self.data_path['results'] = 'ours.csv'
        self.data_path['model_save'] = 'model_default.pkl'
        self.data_path['freq_path'] = 'freq.csv'
        
        self.data_path['time_csv'] = 'time.csv'
        self.data_path['time_pkl'] = 'time.pkl'

        self.data_path.update(data_path)

        self.n_jobs = n_jobs
        self.verbose = verbose
        self.refit = refit
        self.n_splits = n_splits
        self.return_train_score = return_train_score
        self.whether_record = whether_record
        
        self.X, self.Y = np.load(self.data_path['tot_X']), np.load(self.data_path['tot_Y'])

        self.cv = None

        self.scoring = scorer
        
        self.cv_results_ = None
        self.best_estimator_ = None # type = list, not record the best estimator, yet record the estimator under the best params
        self.best_params_ = None  
        self.best_index_ = None
        self.mean_train_time = None
        self.best_on_val = []
        self.best_on_test = []
        
    def scaler_norm(self,train_X,test_X,val_X):
        
        scaler=Normalizer()
        new_train_X=scaler.fit_transform(train_X)
        new_test_X=scaler.transform(test_X)
        new_val_X=scaler.transform(val_X)
        return new_train_X,new_test_X,new_val_X

    def load_test_data(self):
        '''
        This function is used for loading all test data on all split according to their indices.

        Return:
            a list of tuple, [(testX0, testY0), ...., (testX_K, testY_K)]
        '''
        test = []
        for path_x, path_y in self.data_path['test_ids']:
            x = np.load(os.path.sep.join([self.data_path['base_path'], path_x]))
            y = np.load(os.path.sep.join([self.data_path['base_path'], path_y]))
            test.append((self.X[x], self.Y[y]))
        return test 

    def _fit_and_score( self, 
                        estimator, 
                        params, 
                        train, 
                        val,  
                        test,
                        ids):
        trainX, trainY = self.X[train], self.Y[train]
        valX, valY = self.X[val], self.Y[val]
        testX, testY = self.X[test], self.Y[test]

        trainX,testX,valX=self.scaler_norm(trainX,testX,valX)
        
        cv_results = cross_validate(
            estimator = estimator, # model
            trainX = trainX, # train/val data
            trainy = trainY, # train/val data label
            valX = valX,
            valy=valY,
            scoring = self.scoring, # scorers
            verbose = self.verbose, # display message
            fit_params = params, # Params to pass to the fit method of the estimator
            return_train_score = self.return_train_score, # scores on train data,
            return_estimator = True, # return the fitted estimator on each split
            error_score = 'raise' # raise the error, or numerical
        )

        cv_results['left_test_score'] = np.array(auc_m(testY, cv_results['estimator'].predict_proba(testX)))

        cv_results['index'] = ids
        
        if not self.return_train_score:
            return  (cv_results['index'],
                    cv_results['estimator'], 
                    cv_results['test_score'], 
                    cv_results['left_test_score'], 
                    cv_results['fit_time'], 
                    cv_results['score_time'])
        else:
            return (cv_results['index'],
                    cv_results['estimator'], 
                    cv_results['train_score'],
                    cv_results['test_score'], 
                    cv_results['left_test_score'], 
                    cv_results['fit_time'], 
                    cv_results['score_time'])
        
    def fit(self):
        '''
        slef.fit() to start CV
        '''

        base_estimator = clone(self.estimator)
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)

        results = {}
        with parallel:
            all_condidate_params = []
            all_ret = []

            def evaluate_candidates(candidate_params):
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)
                
                out = parallel(delayed(self._fit_and_score)(clone(base_estimator),
                                                        params = params,
                                                        train = train,
                                                        val = val,
                                                        test = test,
                                                        ids = ids)
                            for params, (train, val, test, ids) in product(candidate_params,
                            zip(self.data_path['train_ids'], 
                                self.data_path['val_ids'], 
                                self.data_path['test_ids'], 
                                range(self.n_splits))))

                if len(out) < 1:
                    raise ValueError('No fits were performed !!!')
                elif len(out) != n_candidates * self.n_splits:
                    raise ValueError('Parallel results are inconsistent with our expections!. Expecte {} splits, got {}'
                                    .format(self.n_splits, len(out) // n_candidates))

                all_condidate_params.extend(candidate_params)
                all_ret.extend(out)

                nonlocal results
                results = self.__format__results(all_condidate_params, all_ret)

                return results

            evaluate_candidates(ParamGrid(self.param_grid))

        
        self.cv_results_ = results
        
        self.cur_log.info(self.cv_results_)

        if 'rank_val_score' not in self.cv_results_:
            raise ValueError('rank_val_score not exists in results, can not ranking!' )
        else:
            self.best_index_ = results['rank_val_score'].argmin()
            self.best_params_ = results['params'][self.best_index_]
            self.best_estimator_ = results['estimator'][self.best_index_]
            self.mean_train_time = results['mean_train_fit_time'][self.best_index_]

            self.cur_log.info("best index:")
            self.cur_log.info(self.best_index_)
            self.cur_log.info("best params:")
            self.cur_log.info(self.best_params_)
            
            for i in range(self.n_splits):
                self.best_on_val.append(results['split%d_%s' % (i, 'val_score')][self.best_index_])
                self.best_on_test.append(results['split%d_%s' % (i, 'test_score')][self.best_index_])
            self.cur_log.info("best_on_val:")
            self.cur_log.info(self.best_on_val)
            self.cur_log.info("best_on_test:")
            self.cur_log.info(self.best_on_test)
            self.cur_log.info("mean train time {}".format(self.mean_train_time))


        # save all results during cv process.
        if self.whether_record:
           
            _results = self.cv_results_
            _results['dataname'] = self.dataname
            _results = pd.DataFrame(_results)
            print(_results)
            path = os.path.sep.join([self.data_path['results']])
            _results.to_csv(path, mode = 'a')
            self.cur_log.info("ours results are saved at %s" % (self.data_path['results']))

        self.save_data()
        self.save_time()

    def save_time(self):

        if not os.path.exists(os.path.sep.join([self.data_path['time_pkl']])):
            pkl_data = {self.dataname : {self.method: self.mean_train_time}}
        else:
            with open(os.path.sep.join([self.data_path['time_pkl']]), 'rb') as f:
                pkl_data = pk.load(f, encoding = 'utf-8')
            if self.dataname in pkl_data:
                if not self.method in pkl_data[self.dataname]:
                    pkl_data[self.dataname][self.method] = self.mean_train_time
            else:
                pkl_data[self.dataname] = {self.method: self.mean_train_time}

        with open(os.path.sep.join([self.data_path['time_pkl']]), 'wb') as f:
            pk.dump(pkl_data, f, pk.HIGHEST_PROTOCOL)

        # save to csv
        if not os.path.exists(os.path.sep.join([self.data_path['time_pkl']])):
            raise ValueError("Cannot find pkl file to generate csv!!!")
        
        with open(os.path.sep.join([self.data_path['time_pkl']]), 'rb') as f:
            pkl_data = pk.load(f, encoding = 'utf-8')
        
        data = {}
        for dataname in pkl_data:
            for method in pkl_data[dataname]:
                if method not in data:
                    data[method] = {dataname: pkl_data[dataname][method]}
                else:
                    data[method][dataname] = pkl_data[dataname][method]

        data = pd.DataFrame(data)
        data.to_csv(os.path.sep.join([self.data_path['time_csv']]))

    def __format__results(self, candidate_params, out):
        '''
        (cv_results['estimator'], 
                    cv_results['train_score'],
                    cv_results['test_score'], 
                    cv_results['left_test_score'], 
                    cv_results['fit_time'], 
                    cv_results['score_time'])
        '''
        #print(*out)

        n_candidates = len(candidate_params)
        if self.return_train_score:
            (index_lists, estimator_lists, train_score_lists, val_score_lists, test_score_lists, 
                fit_time_lists, score_time_lists) = zip(*out)
        else:
            (index_lists, estimator_lists, val_score_lists, test_score_lists, 
                fit_time_list, score_time_lists) = zip(*out)
        
        results = {}
        # discard the estimators
        def _store_data(name, array, splits = False, means = False, ranking = False, weights = None, whether_estimator = False): # new added whether_estimator
            '''
            A function to store the CV data like cv_results_ in GridSearchCV 
                
            Params:
                name: 
                    name stored in dicts
                array:
                    array that u wanna store
                means:
                    whether to calculate the average of data
                weights:
                    weighted average
                split:
                    store message for each split
                ranking:
                    rank for selecting the best estimator
                whether_estimator: // disused !!!
                    True: to process and save the estimator on each split.

            Return:
                None
            '''
            if whether_estimator:
                array = np.array(array).reshape(n_candidates, self.n_splits)
            else:
                array = np.array(array, dtype = np.float64).reshape(n_candidates, self.n_splits)

            if splits:
                for i in range(self.n_splits):
                    results['split%d_%s'
                            % (i, name)] = array[:, i]
                # means
            if means:
                array_means = np.average(array, axis = 1, weights=weights)
                results['mean_%s' % name] = array_means
                    # vars
                array_stds = np.sqrt(np.average((array - 
                                                array_means[:, np.newaxis])**2,
                                                axis = 1, weights = weights))
                results['std_%s' % name] = array_stds
            
            # rank for means of val data
            if ranking and means:
                results['rank_%s' % name] = np.asarray(
                    rankdata(-array_means, method = 'min'), dtype = np.int32)
            
        _store_data('train_fit_time', fit_time_lists, splits = True, means = True)
        _store_data('val_score_time', score_time_lists, splits = True, means = True)
        _store_data('val_score', val_score_lists, splits = True, ranking = True, means = True)
        _store_data('test_score', test_score_lists, splits = True, means = True)
        #_store_data('estimator', estimator_lists, splits = True, whether_estimator = True)

        if self.return_train_score:
            _store_data('train_score', train_score_lists, splits = True, means = True)

        param_results = defaultdict(partial(MaskedArray,
                                            np.empty(n_candidates,),
                                            mask=True,
                                            dtype=object))
        for param_i, params in enumerate(candidate_params):
            for name, value in params.items():
                param_results['param_%s' % name][param_i] = value
            
        results['params'] = candidate_params
        results.update(param_results)

        results['estimator'] = np.array(estimator_lists).reshape(n_candidates, self.n_splits)

        return results

    def save_data(self):

        # save the best param for self.method on dataname with corresponding self.n_splits estimators
        
        # save the best param for self.method on dataname with corresponding self.n_splits estimators
        
        with open(os.path.sep.join(['best_param.txt']), 'a') as f:
            print('\n param of ' +  self.method + ' on ' + self.dataname + ':\n', file = f)
            f.write(','.join([str(k) + ' = ' + str(v) for k,v in sorted(self.best_params_.items(), key = lambda x: x[0])]))

        # save the score of test data for 15 splititon with the self.best_params_
        if not os.path.exists(os.path.sep.join([self.data_path['pkl']])):
            pkl_data = {self.dataname : {self.method: self.best_on_test}}
        else:
            with open(os.path.sep.join([self.data_path['pkl']]), 'rb') as f:
                pkl_data = pk.load(f, encoding = 'utf-8')
            if self.dataname in pkl_data:
                if not self.method in pkl_data[self.dataname]:
                    pkl_data[self.dataname][self.method] = self.best_on_test
            else:
                pkl_data[self.dataname] = {self.method: self.best_on_test}

        with open(os.path.sep.join([self.data_path['pkl']]), 'wb') as f:
            pk.dump(pkl_data, f, pk.HIGHEST_PROTOCOL)

        # if u wanna generate csv table
        self.save_to_csv() #np.mean(SCORES), np.std(SCORES)
        
    def save_to_csv(self):

        '''
        # This must be called after all CV experiments finished, and it will produce a whole result table.
        form:
                method1,    method2,    .... methodk
        data1   [mean, std] ...
        data2
        ....
        datak
        '''

        if not os.path.exists(os.path.sep.join([self.data_path['pkl']])):
            raise ValueError("Cannot find pkl file to generate csv!!!")
        
        with open(os.path.sep.join([self.data_path['pkl']]), 'rb') as f:
            pkl_data = pk.load(f, encoding = 'utf-8')
        
        data = {}
        for dataname in pkl_data:
            for method in pkl_data[dataname]:
                mean, std = np.mean(pkl_data[dataname][method]), np.std(pkl_data[dataname][method])

                if method not in data:
                    data[method] = {dataname: [mean, std]}
                else:
                    data[method][dataname] = [mean, std]

        data = pd.DataFrame(data)
        data.to_csv(os.path.sep.join([self.data_path['csv']]))

if __name__ == '__main__':
    
    cv = CrossValidation(
        dataname = 'zoo',
        method = 'focal_loss',
        data_path={'base_path': 'data', 'tot_X': 'X.npy', 'tot_Y': 'Y.npy', 
                'train_ids': 'train_ids.npy', 'val_ids': 'val_ids.npy', 'test_ids': 'test_ids.npy',
                'pkl': 'results.pkl',
                'csv': 'result_table.csv',
                'results': 'ours.csv'
                }
        
    )