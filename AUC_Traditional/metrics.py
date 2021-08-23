import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

def auc_m(y_true, y_pred, label1=None, label2=None, freq = False):
    y_true=np.reshape(y_true,[-1,1])
    enc1 = OneHotEncoder()
    enc1.fit(y_true)
    y_true = enc1.transform(y_true).toarray()
    y_pred_shape=np.shape(y_pred)
    if len(y_pred_shape)==1 or y_pred_shape[1]==1:
        y_pred = np.reshape(y_pred, [-1, 1])
        y_pred = enc1.transform(y_pred).toarray()
    def auc_binary(i, j):
        msk = np.logical_or(y_true.argmax(axis=1) == i, y_true.argmax(axis=1) == j)
        return roc_auc_score(y_true[:, i][msk], y_pred[:, i][msk])
    n = y_true.shape[1]
    
    if not freq:
        return np.mean([auc_binary(i, j) for i in range(n) for j in range(n) if i != j])
    else:
        return auc_binary(label1, label2)
        
