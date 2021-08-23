import abc
from sklearn.base import BaseEstimator, ClassifierMixin
import sklearn
import timeit
import random
import numpy as np
import cfunc
from sklearn import metrics


def timeitfun(fun, var):
    start = timeit.default_timer()
    res = fun(var)
    end = timeit.default_timer()
    diff = end - start
    return res, diff

def toVec(res):
    return res if len(res.shape) == 1 else res.squeeze()


def toMat(res):
    return res if len(res.shape) == 2 else res[:, np.newaxis]


def softmax(Input):
    Output = np.exp(Input)
    Outsum = Output.sum(1)[:, np.newaxis]
    Output = Output / Outsum
    return Output


def toOneHot(Input):
    n = Input.shape[0]
    c = np.max(Input) + 1
    Res = np.zeros((n, c))
    rowindex = np.array(range(n))
    rowindex = rowindex
    colindex = Input.squeeze()
    Res[rowindex, colindex] = 1
    return Res


def cal_info(W, InX, InY):
    InY = InY.astype(int)
    pred = InX.dot(W)
    Delta = softmax(pred)
    n_class = int(np.max(InY) + 1)
    Nclass = np.array([(InY == ind).sum() for ind in range(n_class)]).astype(int)

    D = 1 / Nclass[InY]
    return Delta, Nclass, D


def findNonZeros(x1, x2, thresh, _cython=False):
    assert type(x1) is np.ndarray
    assert type(x2) is np.ndarray

    if len(x1.shape) != 2:
        x1 = x1[np.newaxis, :]
    if len(x2.shape) != 2:
        x2 = x2[np.newaxis, :]

    if _cython:
        return cfunc.findNonZeros(x1, x2, thresh)

    n = len(x1)
    m = len(x2)
    nindex = np.zeros((n, 1))
    p = 0
    q = 0
    while q < m and p < n:
        if (x1[p] - x2[q]) < thresh:
            q += 1
        else:
            nindex[p] = q - 1 if q > 1 else 0
            p += 1
    if q == m:
        nindex[p] = q - 1 if q > 1 else 0
    if p != n:
        nindex[p:] = nindex[p]
    return nindex.astype(int).squeeze()


def calCumSumLoss(Sx, Sy, nindex, DiN, tresh, _cython=False):
    if len(Sx.shape) != 2:
        Sx = Sx[np.newaxis, :]
    if len(Sy.shape) != 2:
        Sy = Sy[np.newaxis, :]

    if _cython:
        return cfunc.calCumSumLoss(Sx, Sy, nindex, DiN, tresh)

    n, m = Sx.shape[0], Sy.shape[0]
    Syx = DiN * Sy
    offset = 0
    w0 = 0
    z0 = 0
    k = n
    deltaX = np.zeros((n, 1))
    DeltaX = np.zeros((n, 1))
    loc = 0
    for i in range(n):
        if nindex[i] != (offset - 1):
            end = nindex[i] + 1
            deltaX[i] = w0 + (DiN[offset:end].sum())
            DeltaX[i] = z0 + (Syx[offset:end].sum())
            offset = end
            w0 = deltaX[i]
            z0 = DeltaX[i]
            loc = i
        else:
            if nindex[i] == m - 1:
                break
            else:
                deltaX[i] = w0
                DeltaX[i] = z0
                loc = i
    if loc < n - 1:
        deltaX[loc:] = w0
        DeltaX[loc:] = z0

    return ((tresh - Sx).transpose().dot(deltaX) + DeltaX.sum()).squeeze()


def calCumSumGrad(deltaGradX, deltaGradY, Sx, Sy, nindex, DiN, _cython=False):
    if len(Sx.shape) != 2:
        Sx = Sx[np.newaxis, :]
    if len(deltaGradX.shape) != 2:
        deltaGradX = deltaGradX[np.newaxis, :]
    if _cython:
        return cfunc.calCumSumGrad(deltaGradX, deltaGradY, Sx, Sy, nindex, DiN)

    n, m = deltaGradX.shape[0], deltaGradY.shape[0]
    d = Sx.shape[1]
    nc = deltaGradX.shape[1]
    deltaGradYx = DiN * deltaGradY
    offset = 0
    w0 = 0
    k = n
    gamma_x = np.zeros((n, 1))
    Gamma_x = np.zeros((d, nc))
    loc = 0
    k = n
    for i in range(n):
        if nindex[i] != (offset - 1):
            end = int(nindex[i].squeeze() + 1)
            gamma_x[i] = w0 + (DiN[offset:end].sum())
            Gamma_x += k * Sy[offset:end, :].transpose().dot(deltaGradYx[offset:end, :])
            k -= 1
            offset = end
            w0 = gamma_x[i]
            loc = i
        else:
            if nindex[i] == m - 1:
                break
            else:
                gamma_x[i] = w0
                loc = i
                k -= 1
    if loc < n - 1:
        gamma_x[loc:] = w0

    return (-Sx).transpose().dot(gamma_x * deltaGradX) + Gamma_x


def calsoftgrad(delta, i):
    
    res = -delta
    res[:, i] = res[:, i] + 1
    di = delta[:, i]
    di = di[:, np.newaxis]
    return di * res

class AUCMLin(BaseEstimator, metaclass=abc.ABCMeta):
    def __init__(self, lambda1, numiter, iterinner, thresh, stepsize, acc=True, verbose=True):
        super().__init__()
        self.lambda1 = lambda1
        self.numiter = numiter
        self.W = None
        self.num_feature = None
        self.num_class = None
        self.thresh = thresh
        self.merge = True
        self.stepsize = stepsize
        self.verbose = verbose
        self.iterinner = iterinner
        self.X = None
        self.Y = None
        self.acc = acc
        self._set_evaluators()

    def _get_params(self, deep=True):
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def _set_evaluators(self):
        self.loss_evaluator = self._cal_loss_per_class if self.acc else self._cal_loss_per_class_easy
        self.grad_evaluator = self._cal_grad_per_class if self.acc else self._cal_grad_per_class_easy

    def _set_params(self, **params):
        for key, value in params.items():
            if key == 'lambda1':
                self.lambda1 = value
            elif key == 'numiter':
                self.numiter = value
            elif key == 'iterinner':
                self.iterinner = value
            elif key == 'thresh':
                self.thresh = value
            elif key == 'stepsize':
                self.stepsize = value
            elif key == 'acc':
                self.acc = 'acc'

    def _call_linear_approx(self, W_x, W_y, alpha):
        fy = self.cal_loss(W_y)
        g_y = self.cal_grad(W_y)
        g_y = g_y.reshape(-1)
        dW = (W_x - W_y).reshape(-1)
        return fy + g_y.dot(dW) + 1 / (2 * alpha) * (np.linalg.norm(dW) ** 2)

    def _BBstepsize(self, W_x, W_yold):
        s = (W_x - W_yold).reshape(-1)
        r = (self.cal_grad(W_x) - self.cal_grad(W_yold)).reshape(-1)
        a = s.dot(s)
        b = s.dot(r)

        return a / (b + 0.01 * a) if b + 0.01 * a > 0.01 else self.stepsize

    def _linesearch(self, W_x, W_y, W_y_old, iterinner):

        gradY = self.cal_grad(W_y)
        stepsize = self._BBstepsize(W_x, W_y_old)
        for j in range(iterinner):
            W = W_y - stepsize * gradY
            if self.cal_loss(W) > self._call_linear_approx(W, W_y, stepsize):
                stepsize /= 1.5
            else:
                break
        return W

    def fit(self, xX, xY):
        self.X = xX
        self.Y = xY.astype(int)

        if np.min(self.Y) == 1:
            self.Y -= 1
        self.num_feature = self.X.shape[1]
        self.num_class = int(np.max(self.Y).squeeze() + 1)
        self.W = np.zeros((self.num_feature, self.num_class))
        _, self.Nclass, self.D = cal_info(self.W, self.X, self.Y)
        n = self.X.shape[0]
        
        W_x = np.zeros((self.num_feature, self.num_class))

        W_x_old = np.zeros((self.num_feature, self.num_class))
        
        W_y = np.zeros((self.num_feature, self.num_class))
        W_y_old = np.zeros((self.num_feature, self.num_class))
        
        W_z = np.zeros((self.num_feature, self.num_class))

        W_v = np.zeros((self.num_feature, self.num_class))
        loss = []
        t = 1.0
        t_old = 0
        stepsize_z = self.stepsize
        stepsize_v = self.stepsize
        timeTie = 0
        for iter in range(self.numiter):
            W_y_old = W_y.copy()
            W_y = W_x + (t_old / t) * (W_z - W_x) + ((t_old - 1) / t) * (W_x - W_x_old)
            W_z = self._linesearch(W_z, W_y, W_y_old, self.iterinner)
            W_v = self._linesearch(W_v, W_x, W_x_old, self.iterinner)

            t_old = t
            t = (1.0 + np.sqrt(1.0 + 4 * t ** 2)) / 2
            W_x_old = W_x.copy()
            l1, l2 = self.cal_loss(W_z), self.cal_loss(W_v)
            W_x = W_v if l1 > l2 else W_z

            if self.verbose:
                loss.append(np.min([l1, l2]))
                print('loss of the {:.4f} iteration: {:.4f} '.format(iter, loss[-1]))

                if len(loss) >= 2 and abs(loss[-1] - loss[-2]) < 1e-5:
                    timeTie += 1
                else:
                    timeTie = 0
                if timeTie == 5:
                    break
        self.W = W_x
        return self

    def compare_speed(self, Xx, Yy, oneTime=False, needloss=False):

        if oneTime:
            self.X = Xx
            self.Y = Yy.astype(int)
            n, d = self.X.shape
            self.num_class = np.max(self.Y).astype(int) + 1
            W = np.random.rand(d, self.Nclass)
            _, self.Nclass, self.D = cal_info(W, Xx, Yy)
            self.W = W

            self.acc = False
            self._set_evaluators()
            l1, time1l = timeitfun(self.cal_loss, self.W)
            g1, time1g = timeitfun(self.cal_grad, self.W)

            self.acc = True
            self._set_evaluators()
            l2, time2l = timeitfun(self.cal_loss, self.W)
            g2, time2g = timeitfun(self.cal_grad, self.W)
            if needloss:
                return np.abs(l1 - l2), np.sum((g1 - g2) ** 2), time1g / time2g, time1l / time2l
            return time1g / time2g, time1l / time2l
        else:
            self.acc = False
            self._set_evaluators()
            start = timeit.default_timer()
            self.fit(Xx, Yy)
            endt = timeit.default_timer()
            time1 = endt - start

            self.acc = True
            self._set_evaluators()
            start = timeit.default_timer()
            self.fit(Xx, Yy)
            endt = timeit.default_timer()
            time2 = endt - start
            return time1, time2

    @abc.abstractmethod
    def _cal_loss_per_class(self, delta, Di, i):
        pass

    @abc.abstractmethod
    def _cal_grad_per_data(self, varin):
        pass

    @abc.abstractmethod
    def _cal_loss_per_data(self, varin):
        pass

    def _cal_loss_per_class_easy(self, delta, Di, i):
        X = self.X
        Y = toVec(self.Y)
        n_p, n_n = sum(Y == i), sum(Y != i)
        deltap, deltan = delta[Y == i], delta[Y != i]
        Din = Di[Y != i]

        loss = 0
        for ip in range(n_p):
            for iN in range(n_n):
                loss += Din[iN] * self._cal_loss_per_data(deltap[ip] - deltan[iN])
        return loss

    def _cal_grad_per_class_easy(self, delta, Di, deltaGrad, i):
        X = self.X
        Y = toVec(self.Y)
        delta = toVec(delta)
        n_p, n_n = sum(Y == i), sum(Y != i)

        Xp, Xn = X[Y == i, :], X[Y != i, :]
        deltap, deltan = delta[Y == i], delta[Y != i]
        deltaGradp, deltaGradn = deltaGrad[Y == i, :], deltaGrad[Y != i, :]

        Din = Di[Y != i]

        grad = np.zeros((self.W.shape))
        for ip in range(n_p):
            for iN in range(n_n):
                grad_l = self._cal_grad_per_data(deltap[ip] - deltan[iN])
                # if (grad_l == 0):
                #     print('zero')
                if (grad_l != 0):
                    grad1 = toMat(Xp[ip, :]).dot(toMat(deltaGradp[ip, :]).transpose())
                    grad2 = toMat(Xn[iN, :]).dot(toMat(deltaGradn[iN, :]).transpose())
                    grad += Din[iN] * grad_l * (grad1 - grad2)

        return grad

    @abc.abstractmethod
    def _cal_grad_per_class(self, delta, Di, deltagrad, i):
        pass

    def cal_loss(self, W):
        delta = self.predict_w(W)
        vW = W.reshape(-1)
        return np.sum([self.loss_evaluator(delta[:, i], self.D / self.Nclass[i], i) for i in range(self.num_class)]) + (
                self.lambda1 / 0.5) * (vW.dot(vW))

    def cal_grad(self, W):
        delta = self.predict_w(W)
        for i in range(self.num_class):
            Di = self.D / self.Nclass[i]
            deltaGrad = calsoftgrad(delta, i)
            if i == 0:
                grad = self.grad_evaluator(delta[:, i], Di, deltaGrad, i)
            else:
                grad += self.grad_evaluator(delta[:, i], Di, deltaGrad, i)

        if self.merge and self.acc:
            grad = self.X.transpose().dot(grad)
        ## to be changed
        return grad + self.lambda1 * W

    def predict_w(self, W):
        return softmax(self.X.dot(W))

    def predict(self, X):
        return softmax(X.dot(self.W))

    def predict_proba(self, X):
        return softmax(X.dot(self.W))

    def prdict_proba(self, X):
        return softmax(self.predict(X))


class expAUCMLin(AUCMLin):

    def __init__(self, lambda1, numiter, iterinner, thresh, stepsize, verbose=True):
        super().__init__(lambda1, numiter, iterinner, thresh, stepsize, verbose)
        self.merge = True

    def _cal_loss_per_class(self, delta, Di, i):
        delta = delta[:, np.newaxis]
        return ((self.Y == i) * np.exp(-self.thresh * delta)).sum() * (
                (self.Y != i) * Di * np.exp(self.thresh * delta)).sum()

    def _cal_grad_per_class(self, delta, Di, deltagrad, i):
        delta = delta[:, np.newaxis]
        nc = deltagrad.shape[1]
        Delta1 = np.exp(-self.thresh * delta)
        Delta2 = np.exp(self.thresh * delta)
        l1 = ((self.Y == i) * Delta1).sum()
        l2 = ((self.Y != i) * Di * Delta2).sum()
        C1 = l2 * (self.Y == i) * -Delta1
        C2 = l1 * (self.Y != i) * Di * Delta2
        return self.thresh * (C1 + C2) * deltagrad

    def _cal_loss_per_data(self, varin):
        return np.exp(-self.thresh * varin)

    def _cal_grad_per_data(self, varin):
        return -self.thresh * np.exp(-self.thresh * varin)


class HingeAUCMLin(AUCMLin):
    def __init__(self, lambda1, numiter, iterinner, thresh, stepsize, verbose=True, _cython=False):
        super().__init__(lambda1, numiter, iterinner, thresh, stepsize, verbose)
        self.merge = False
        self._cython = _cython

    def _cal_loss_per_class(self, delta, Di, i):
        indexI = (self.Y == i).squeeze()
        indexIN = (self.Y != i).squeeze()
        
        delta = delta[:, np.newaxis]
        deltaP = delta[indexI]
        deltaN, DiN = delta[indexIN], Di[indexIN]
        indexsortP = deltaP.argsort(0)[::-1].squeeze()
        indexsortN = deltaN.argsort(0)[::-1].squeeze()
        deltaP = deltaP[indexsortP]
        deltaN, DiN = deltaN[indexsortN], DiN[indexsortN]
        
        nindex = findNonZeros(deltaP, deltaN, self.thresh, self._cython)

        return calCumSumLoss(deltaP, deltaN, nindex, DiN, self.thresh, _cython=self._cython)

    def _cal_grad_per_class(self, delta, Di, deltagrad, i):
        indexI = (self.Y == i).squeeze()
        indexIN = (self.Y != i).squeeze()
        
        delta = delta[:, np.newaxis]
        deltaP, deltagradP, Xp = delta[indexI], deltagrad[indexI, :], self.X[indexI, :]

        deltaN, deltagradN, XN, DiN = delta[indexIN], deltagrad[indexIN, :], self.X[indexIN, :], Di[indexIN]
        indexsortP = deltaP.argsort(0)[::-1].squeeze()
        indexsortN = deltaN.argsort(0)[::-1].squeeze()
        deltaP, deltagradP, Xp = deltaP[indexsortP], deltagradP[indexsortP, :], Xp[indexsortP, :]
        deltaN, DiN, deltagradN, XN = deltaN[indexsortN], DiN[indexsortN], deltagradN[indexsortN, :], XN[indexsortN,
                                                                                                      :]
        nindex = findNonZeros(deltaP, deltaN, self.thresh, self._cython)

        return calCumSumGrad(deltagradP, deltagradN, Xp, XN, nindex, DiN, self._cython)

    def _cal_loss_per_class_naive(self, delta, Di, i):
        delta = delta[:, np.newaxis]
        Ni = (self.Y == i).sum()
        s1 = ((self.Y == i) * ((self.num_class - 1) / Ni) * (self.thresh - delta)).sum()
        s2 = ((self.Y != i) * Ni * Di * delta).sum()
        return s1 + s2

    def _cal_grad_per_class_naive(self, delta, Di, deltagrad, i):
        delta = delta[:, np.newaxis]
        Ni = (self.Y == i).sum()
        s1 = ((self.Y == i) * ((self.num_class - 1) / Ni) * (- deltagrad))
        s2 = ((self.Y != i) * Ni * Di * deltagrad)
        return self.X.transpose().dot(s1 + s2)

    def _cal_loss_per_data(self, varin):
        return self.thresh - varin if self.thresh > varin else 0

    def _cal_grad_per_data(self, varin):
        return -1 if self.thresh > varin else 0


class LSAUCMLin(AUCMLin):
    def __init__(self, lambda1, numiter, iterinner, thresh, stepsize, verbose=True):
        super().__init__(lambda1, numiter, iterinner, thresh, stepsize, verbose)
        self.merge = True

    def _cal_loss_per_class(self, delta, Di, i):
        Ni = np.sum(self.Y == i)
        Di = Di.squeeze()
        Yi = (self.Y == i).squeeze()
        
        diff = delta - self.thresh * Yi
        nD = (Di * (1 - Yi))
        S = Ni * nD + ((self.num_class - 1) * Yi / Ni)
        A = ((diff * S)).dot(diff)
        B = diff.dot(nD) * Yi.dot(diff)
        return 0.5 * A - B

    def _cal_grad_per_class(self, delta, Di, deltagrad, i):
        Ni = np.sum(self.Y == i)
        delta = delta[:, np.newaxis]
        Yi = self.Y == i
       
        diff = delta - self.thresh * Yi
        nD = Di * (1 - Yi)
        w = Ni * nD + ((self.num_class - 1) * Yi / Ni)
        grad1 = diff * w * deltagrad
        l1 = diff.transpose().dot(Yi)
        l2 = nD.transpose().dot(diff)
        grad2 = l2 * Yi * deltagrad + l1 * nD * deltagrad

        return grad1 - grad2

    def _cal_loss_per_data(self, varin):
        return 0.5 * (self.thresh - varin) ** 2

    def _cal_grad_per_data(self, varin):
        return varin - self.thresh

def get_acc_ratio(datas, model):
    return np.array(
        [np.array([model.compare_speed(dataX, Yy, True) for _ in range(5)]).mean(0) for (dataX, Yy) in datas])