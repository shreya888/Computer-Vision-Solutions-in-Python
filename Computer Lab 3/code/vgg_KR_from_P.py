# -*- coding: utf-8 -*-
'''
%VGG_KR_FROM_P Extract K, R from camera matrix.
%
%    [K,R,t] = VGG_KR_FROM_P(P [,noscale]) finds K, R, t such that P = K*R*[eye(3) -t].
%    It is det(R)==1.
%    K is scaled so that K(3,3)==1 and K(1,1)>0. Optional parameter noscale prevents this.
%
%    Works also generally for any P of size N-by-(N+1).
%    Works also for P of size N-by-N, then t is not computed.

% original Author: Andrew Fitzgibbon <awf@robots.ox.ac.uk> and awf
% Date: 15 May 98

% Modified by Shu.
% Date: 8 May 20
'''
import numpy as np

def vgg_rq(S):
    S = S.T
    [Q,U] = np.linalg.qr(S[::-1,::-1], mode='complete')

    Q = Q.T
    Q = Q[::-1, ::-1]
    U = U.T
    U = U[::-1, ::-1]
    if np.linalg.det(Q)<0:
        U[:,0] = -U[:,0]
        Q[0,:] = -Q[0,:]
    return U,Q


def vgg_KR_from_P(P, noscale = True):
    N = P.shape[0]
    H = P[:,0:N]
    print(N,'|', H)
    [K,R] = vgg_rq(H)
    if noscale:
        K = K / K[N-1,N-1]
        if K[0,0] < 0:
            D = np.diag([-1, -1, np.ones([1,N-2])]);
            K = K @ D
            R = D @ R
        
            test = K*R; 
            assert (test/test[0,0] - H/H[0,0]).all() <= 1e-07
    
    t = np.linalg.inv(-P[:,0:N]) @ P[:,-1]
    return K, R, t


# Load both the calibration matrices
P = np.load('camera_calib.npy')
P_resize = np.load('camera_calib_resize.npy')

# Decompose P into K, R, and t such that P = K[R|t]
K, R, t = vgg_KR_from_P(P)
print('\nDecomposing P into K, R, t for I image:')
print('K = \n', K)
print('R = \n', R)
print('t = \n', t)
print()

# Decompose P into K, R, and t such that P = K[R|t]
K_resize, R_resize, t_resize = vgg_KR_from_P(P_resize)
print('\nDecomposing P into K\', R\', t\' for I_resize image:')
print('K\' = \n', K_resize)
print('R\' = \n', R_resize)
print('t\' = \n', t_resize)

#K, R, t = vgg_KR_from_P(C)
print()
