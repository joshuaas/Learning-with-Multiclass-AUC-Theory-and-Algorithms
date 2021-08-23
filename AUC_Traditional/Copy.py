__authors__ = "Statusrank"
__copyright__ = 'baoshilong@iie.com'

import copy
import numpy as np 
import time
def clone(estimator, safe=True):
    """Constructs a new estimator with the same parameters.
    Clone does a deep copy of the model in an estimator
    without actually copying attached data. It yields a new estimator
    with the same parameters that has not been fit on any data.
    Parameters
    ----------
    estimator : estimator object, or list, tuple or set of objects
        The estimator or group of estimators to be cloned
    safe : bool, default=True
        If safe is false, clone will fall back to a deep copy on objects
        that are not estimators.
    """
    estimator_type = type(estimator)
    # XXX: not handling dictionaries
    if estimator_type in (list, tuple, set, frozenset):
        return estimator_type([clone(e, safe=safe) for e in estimator])
    elif not hasattr(estimator, 'get_params') or isinstance(estimator, type):
        if not safe:
            return copy.deepcopy(estimator)
        else:
            if isinstance(estimator, type):
                raise TypeError("Cannot clone object. " +
                                "You should provide an instance of " +
                                "scikit-learn estimator instead of a class.")
            else:
                raise TypeError("Cannot clone object '%s' (type %s): "
                                "it does not seem to be a scikit-learn "
                                "estimator as it does not implement a "
                                "'get_params' method."
                                % (repr(estimator), type(estimator)))

    klass = estimator.__class__
    new_object_params = estimator.get_params(deep=False)
    for name, param in new_object_params.items():
        new_object_params[name] = clone(param, safe=False)
    new_object = klass(**new_object_params)
    params_set = new_object.get_params(deep=False)

    # quick sanity check of the parameters of the clone
    for name in new_object_params:
        param1 = new_object_params[name]
        param2 = params_set[name]

        if param1 is not param2:
            raise RuntimeError('Cannot clone object %s, as the constructor '
                                'either does not set or modifies parameter %s' %
                                (estimator, name))
    return new_object

def cross_validate(estimator, trainX, trainy, valX, valy,
                    scoring, fit_params, verbose=0, 
                    return_train_score=False,
                    return_estimator=False, 
                    error_score=np.nan):

    scores = _fit_and_score(estimator, 
            trainX, 
            trainy, 
            valX, 
            valy, 
            scoring,
            fit_params, 
            return_train_score=return_train_score,
            return_times=True, 
            return_estimator=return_estimator,
            error_score=error_score)
    
    if return_train_score:
        train_scores = scores.pop(0)
    if return_estimator:
        fitted_estimators = scores.pop()
    test_scores, fit_times, score_times = scores

    ret = {}
    ret['fit_time'] = np.array(fit_times)
    ret['score_time'] = np.array(score_times)

    if return_estimator:
        ret['estimator'] = fitted_estimators

    
    ret['test_score'] = np.array(test_scores)
    if return_train_score:
        ret['train_score'] = np.array(train_scores)

    return ret

def _fit_and_score(estimator, trainX, trainy, valX, valy, scorer,
                   parameters, 
                   verbose = 1,
                   return_train_score=False,
                   return_parameters=False, 
                   return_times=False, 
                   return_estimator=False,
                   error_score='raise'):
    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.

        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.

        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.

    train : array-like, shape (n_train_samples,)
        Indices of training samples.

    test : array-like, shape (n_test_samples,)
        Indices of test samples.

    verbose : integer
        The verbosity level.

    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised. This parameter
        does not affect the refit step, which will always raise the error.

    parameters : dict or None
        Parameters to be set on the estimator.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    return_train_score : boolean, optional, default: False
        Compute and return score on training set.

    return_parameters : boolean, optional, default: False
        Return parameters that has been used for the estimator.

    return_n_test_samples : boolean, optional, default: False
        Whether to return the ``n_test_samples``

    return_times : boolean, optional, default: False
        Whether to return the fit/score times.

    return_estimator : boolean, optional, default: False
        Whether to return the fitted estimator.

    Returns
    -------
    train_scores : dict of scorer name -> float, optional
        Score on training set (for all the scorers),
        returned only if `return_train_score` is `True`.

    test_scores : dict of scorer name -> float, optional
        Score on testing set (for all the scorers).

    n_test_samples : int
        Number of test samples.

    fit_time : float
        Time spent for fitting in seconds.

    score_time : float
        Time spent for scoring in seconds.

    parameters : dict or None, optional
        The parameters that have been evaluated.

    estimator : estimator object
        The fitted estimator
    """

    train_scores = {}
    if parameters is not None:
        
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator._set_params(**cloned_parameters)

    start_time = time.time()
    
    estimator.fit(trainX, trainy)
    
    fit_time = time.time() - start_time
    test_scores = _score(estimator, valX, valy, scorer)
    score_time = time.time() - start_time - fit_time
    if return_train_score:
        train_scores = _score(estimator, trainX, trainy, scorer)
    

    ret = [train_scores, test_scores] if return_train_score else [test_scores]

    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    if return_estimator:
        ret.append(estimator)
    return ret

def _score(estimator, X_test, y_test, scorer):
    """Compute the score(s) of an estimator on a given test set.
    """
    y_pre = estimator.predict_proba(X_test)
    scores = scorer(y_test, y_pre)
    return scores