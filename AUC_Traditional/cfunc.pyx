cython: infer_types= False

cimport cython
cimport numpy as np
import numpy as np 

from cython.parallel import prange, parallel

ctypedef fused my_type:
    int 
    double 
    long long 

@cython.boundscheck(False)
@cython.wraparound(False)

def findNonZeros(my_type [:, ::1] x1 not None, my_type [:, ::1] x2 not None, double thresh):
    
    cdef int n, m
    if my_type is int:
        dtype = np.intc
    elif my_type is double:
        dtype = np.double
    elif my_type is cython.longlong:
        dtype = np.longlong

    n = x1.shape[0]
    m = x2.shape[0]
    
    cdef np.ndarray[my_type, ndim=2] nindex = np.zeros((n, 1), dtype = dtype)

    cdef my_type[:, ::1] nindex_view = nindex

    cdef Py_ssize_t p, q
    p = 0
    q = 0
    with nogil:
        while q < m and p < n:
            if (x1[p, 0] - x2[q, 0]) < thresh:
                q += 1
            else:
                nindex_view[p] = max(q - 1, 0)
                p += 1
    if q == m:
        nindex_view[p] = max(q - 1, 0)
    if p != n:
        nindex_view[p:] = nindex_view[p]
    return nindex 

@cython.boundscheck(False)
@cython.wraparound(False)

def calCumSumLoss(np.ndarray[my_type, ndim=2] Sx not None, np.ndarray[my_type, ndim=2] Sy not None, np.ndarray[my_type, ndim=2] nindex, np.ndarray[my_type, ndim=2] DiN, double tresh):


    if my_type is int:
        dtype = np.intc
    elif my_type is double:
        dtype = np.double
    elif my_type is cython.longlong:
        dtype = np.longlong

    cdef int offset = 0, end
    cdef my_type w0 = 0, z0 = 0

    cdef Py_ssize_t n = Sx.shape[0], i, loc = 0, m = Sy.shape[0]

    cdef np.ndarray[my_type, ndim=2] Syx = DiN * Sy

    cdef np.ndarray[my_type, ndim=2] deltaX = np.zeros((n, 1), dtype = dtype)
    cdef np.ndarray[my_type, ndim=2] DeltaX = np.zeros((n, 1), dtype = dtype)

    cdef my_type [:, ::1] deltaX_view = deltaX
    cdef my_type [:, ::1] DeltaX_view = DeltaX
    cdef my_type [:, ::1] nindex_view = nindex

    for i in range(n):
        if nindex[i, 0] != (offset - 1):
            end = int(nindex[i, 0] + 1)
            deltaX_view[i, 0] = DiN[offset:end].sum() + w0
            DeltaX_view[i, 0] = Syx[offset:end].sum() + z0
            offset = end
            w0 = deltaX_view[i, 0]
            z0 = DeltaX_view[i, 0]
            loc = i
        else:
            if nindex[i, 0] == m - 1:
                break
            else:
                deltaX_view[i, 0] = w0
                DeltaX_view[i] = z0
                loc = i

    if loc < n - 1:
        deltaX_view[loc:] = w0
        DeltaX_view[loc:] = z0

    #return ((np.tile(tresh, n) - Sx).transpose().dot(deltaX) + DeltaX.sum()).squeeze()
    return ((tresh - Sx).transpose().dot(deltaX) + DeltaX.sum()).squeeze()


@cython.boundscheck(False)
@cython.wraparound(False)
def calCumSumGrad(np.ndarray[my_type, ndim=2] deltaGradX, np.ndarray[my_type, ndim=2] deltaGradY,  np.ndarray[my_type, ndim=2] Sx,  np.ndarray[my_type, ndim=2] Sy,  np.ndarray[my_type, ndim=2] nindex,  np.ndarray[my_type, ndim=2] DiN):
    
    cdef Py_ssize_t n = deltaGradX.shape[0], d = Sx.shape[1], nc = deltaGradX.shape[1], m = deltaGradY.shape[0], k, loc = 0, i
    cdef int offset = 0, end 
    cdef my_type w0 = 0

    if my_type is int:
        dtype = np.intc
    elif my_type is double:
        dtype = np.double
    elif my_type is cython.longlong:
        dtype = np.longlong

    cdef np.ndarray[my_type, ndim=2] deltaGradYx = DiN * deltaGradY
    
    cdef np.ndarray[my_type, ndim=2] gamma_x = np.zeros((n, 1))
    cdef np.ndarray[my_type, ndim=2] Gamma_x = np.zeros((d, nc))
    cdef my_type [:, ::1] gamma_x_view = gamma_x
    cdef my_type [:, ::1] Gamma_x_view = Gamma_x
    k = n
    for i in range(n):
        if nindex[i, 0] != (offset - 1):
            end = int(nindex[i, 0] + 1)
            gamma_x_view[i, 0] = w0 + DiN[offset:end].sum()
            Gamma_x += k * Sy[offset:end].transpose().dot(deltaGradYx[offset:end])
            k -= 1
            offset = end
            w0 = gamma_x_view[i, 0]
            loc = i
        else:
            if nindex[i, 0] == m - 1:
                break
            else:
                gamma_x_view[i, 0] = w0
                loc = i
                k -= 1

    if loc < n - 1:
        gamma_x_view[loc:] = w0

    return (-Sx).transpose().dot(gamma_x * deltaGradX) + Gamma_x