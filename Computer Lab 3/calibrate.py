# -*- coding: utf-8 -*-
# CLAB3 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

I = Image.open('stereo2012d.jpg')

plt.imshow(I)
uv = plt.ginput(6)  # Graphical user interface to get 6 points
print('The uv coordinates of 6 points chosen on image I = ', uv)
#####################################################################
'''
normalize_2d function: Normalize 2D image points using T_norm matrix (formula from lecture slides)
Arguments:
im = image                     - needed to find height and width for T_norm matrix
x  = uv points marked on image - to be normalized using T_norm
Returns:
x      = Normalized uv coordinates of x
T_norm = Matrix used to normalize and denormalize the 2D points in x
'''
def normalize_2d(im, x):
    H, W, _ = im.shape
    T_norm = np.linalg.pinv(np.array([[W + H, 0, W / 2], [0, W + H, H / 2], [0, 0, 1]]))  # formula from lecture slides
    x = np.dot(T_norm, np.concatenate((x.T, np.ones((1, x.shape[0])))))
    x = x.T  # Normalized x
    # Need T_norm matrix later for denormalization
    return x, T_norm

'''
normalize_3d function: Normalize 3D world points using S_norm matrix
Arguments:
X  = XYZ real world points - to be normalized using S_norm
Return:
X      = Normalized XYZ coordinates of X
S_norm = Matrix used to normalize and denormalize the 3D points in X
'''
def normalize_3d(X):
    V_diag_Vinv = 0
    mu = np.mean(X, axis=0)
    for i in range(len(X)):
        V_diag_Vinv += ((np.reshape((X[i] - mu), (-1, 1))) @ (np.reshape((X[i] - mu), (1, -1))))
    diag, V = np.linalg.eig(V_diag_Vinv)
    V_inv = np.linalg.pinv(V)
    diag_inv = np.array([[1 / diag[0], 0, 0], [0, 1 / diag[1], 0], [0, 0, 1 / diag[2]]])
    first = V @ diag_inv @ V_inv
    second = (-V @ diag_inv @ V_inv @ mu).reshape(-1, 1)
    third = np.array([0, 0, 0, 1])
    S = np.append(first, second, axis=1)
    S_norm = np.vstack((S, third))  # formula from lecture slides
    X = S_norm @ np.concatenate((X.T, np.ones((1, X.shape[0]))))
    X = X.T  # Normalized X
    # Need T_norm and S_norm matrices for denormalization later
    return X, S_norm


########################################################################
def calibrate(im, XYZ, uv):
    # Normalize the data in homogenized form
    norm_uv, T_norm = normalize_2d(im, uv)
    norm_XYZ, S_norm = normalize_3d(XYZ)

    # Find A
    A = []
    # Assemble n 12x2 Ai matrices into 2nx12 matrix A
    for i in range(XYZ.shape[0]):
        X, Y, Z = norm_XYZ[i, 0], norm_XYZ[i, 1], norm_XYZ[i, 2]
        u, v = norm_uv[i, 0], norm_uv[i, 1]
        # Compute Ai
        A.append(np.array([[X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u],
                           [0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v]]))
    A = np.array(A)
    A = A.reshape((A.shape[0] * A.shape[1], -1))
    # Compute SVD of A
    U, S, V = np.linalg.svd(A)
    # The solution for P is the last column of V <- smallest singular value of SVD of A
    p = V[-1]
    # Camera projection matrix
    P = p.reshape(3, 4)
    # Normalize P such that P34 = 1
    P = P / P[-1, -1]

    # Denormalization
    # pinv: Moore-Penrose pseudo-inverse of a matrix, generalized inverse of a matrix using its SVD
    # P = T_norm ^ -1 @ P_normalized @ S_norm
    P = np.linalg.pinv(T_norm) @ P @ S_norm
    P = P / P[-1, -1]  # Re-normalize so that P34 = 1

    # Mean squared error between the positions of the uv coordinates and the projected XYZ coordinates
    uv2 = np.dot(P, np.concatenate((XYZ.T, np.ones((1, XYZ.shape[0])))))  # Projected XYZ
    uv2 = uv2 / uv2[2, :]
    err = np.sqrt(np.mean((uv2[:2].T - uv) ** 2))
    print("Error = ", err)

    return P


# XYZ coordinate corresponding to UV coordinates chosen
# XYZ = np.array([[7,7,0], [14,7,0], [0,7,7], [0,14,7], [7,0,7], [7,0,14]], dtype=np.float32)
XYZ = np.array([[7, 0, 0], [7, 7, 0], [0, 7, 0], [0, 7, 7], [0, 0, 7], [7, 0, 7]], dtype=np.float32)
# Convert all to np array
I = np.asarray(I)
uv = np.asarray(uv)

# Visualize points on image
x = uv[:, 0]
y = uv[:, 1]
plt.scatter(x, y, marker='x', c="red", s=20, label='point')
plt.imshow(I)
plt.legend()
plt.title('Visualizing points selected on image')
plt.show()

# Get 3x4 calibration matrix for I image
P = calibrate(I, XYZ, uv)
print('Camera calibration matrix for I image,  P = \n', P)

# Visualize the XYZ coordinate at the origin using calibration matrix P
image_unit_coordi = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])  # unit vectors in XYZ direction in world plane
XYZ_in_image = np.dot(P,
                      np.concatenate((image_unit_coordi.T, np.ones((1, image_unit_coordi.shape[0])))))  # Projected XYZ
XYZ_in_image = XYZ_in_image.T
image_XYZ = np.array([(379.84341391672035, 321.2793297292949), (410.3992931992175, 337.29120508220285),
                      (379.8210237531681, 280.42452919615766), (344.54668295241163, 335.7711207207857)])
x = image_XYZ[:, 0]
y = image_XYZ[:, 1]
plt.plot([x[0], x[1]], [y[0], y[1]], color="blue", linewidth=3, label='X-axiz')
plt.plot([x[0], x[2]], [y[0], y[2]], color="green", linewidth=3, label='Y-axiz')
plt.plot([x[0], x[3]], [y[0], y[3]], color="red", linewidth=3, label='Z-axiz')
plt.legend()
plt.imshow(I)
plt.title('Visualizing XYZ coordinate')
plt.show()

# Save camera calibration matrix
np.save('camera_calib.npy', P)
# Decompose P into K, R, and t such that P = K[R|t] by running vgg_KR_from_P


# Resize image I and display
I_resize = cv2.resize(np.array(I), (0, 0), fx=0.5, fy=0.5)
plt.title('Resized image (H/2, W/2)')
plt.imshow(I_resize)
plt.show()

# Choose 6 uv coordinates on the rescaled image
plt.imshow(I_resize)
# uv_resize = plt.ginput(6)  # Uncomment if want to take new 6 points
uv_resize = uv / 2
print('\nThe uv coordinates of 6 points chosen on resized image = ', uv_resize)
uv_resize = np.asarray(uv_resize)
# Visualize points
x = uv_resize[:,0]
y = uv_resize[:,1]
plt.scatter(x, y, marker='x', c="red", s=20, label='point')
plt.imshow(I_resize)
plt.legend()
plt.title('Visualizing points selected on resized image I_resize')
plt.show()

XYZ_resize = XYZ  # XYZ is same because world coordinate would not change
# Get 3x4 calibration matrix for resized image I_resize
P_resize = calibrate(I_resize, XYZ_resize, uv_resize)
print('Camera calibration matrix for I_resize image,  P\' = \n', P_resize)

# Save camera calibration matrix for resized image
np.save('camera_calib_resize.npy', P_resize)
# Decompose P into K, R, and t such that P = K[R|t] by running vgg_KR_from_P


'''
%% TASK 1: CALIBRATE
%
% Function to perform camera calibration
%
% Usage:   calibrate(image, XYZ, uv)
%          return C
%   Where:   image - is the image of the calibration target.
%            XYZ - is a N x 3 array of  XYZ coordinates
%                  of the calibration target points. 
%            uv  - is a N x 2 array of the image coordinates
%                  of the calibration target points.
%            K   - is the 3 x 4 camera calibration matrix.
%  The variable N should be an integer greater than or equal to 6.
%
%  This function plots the uv coordinates onto the image of the calibration
%  target. 
%
%  It also projects the XYZ coordinates back into image coordinates using
%  the calibration matrix and plots these points too as 
%  a visual check on the accuracy of the calibration process.
%
%  Lines from the origin to the vanishing points in the X, Y and Z
%  directions are overlaid on the image. 
%
%  The mean squared error between the positions of the uv coordinates 
%  and the projected XYZ coordinates is also reported.
%
%  The function should also report the error in satisfying the 
%  camera calibration matrix constraints.
% 
% Shreya Chawla, 22nd May 2021 
'''


############################################################################
def homography(u2Trans, v2Trans, uBase, vBase):
    # The separated coordinates are concatenated
    trans = np.array(list(zip(u2Trans, v2Trans)))
    base = np.array(list(zip(uBase, vBase)))

    # Find A
    A = []
    # Assemble n 2x9 Ai matrices into 2nx9 matrix A
    for i in range(trans.shape[0]):
        x, y = base[i, 0], base[i, 1]
        x_p, y_p = trans[i, 0], trans[i, 1]  # x' and y'
        # Compute Ai
        A.append(np.array([[x, y, 1, 0, 0, 0, -x_p * x, -x_p * y, -x_p],
                           [0, 0, 0, x, y, 1, -y_p * x, -y_p * y, -y_p]]))
    A = np.array(A)
    A = A.reshape((A.shape[0] * A.shape[1], -1))

    U, D, V = np.linalg.svd(A)
    h = V[-1].reshape(3, 3)
    H = h / h[-1, -1]

    return H


'''
%% TASK 2: 
% Computes the homography H applying the Direct Linear Transformation 
% The transformation is such that 
% p = np.matmul(H, p.T), i.e.,
% (uBase, vBase, 1).T = np.matmul(H, (u2Trans , v2Trans, 1).T)
% Note: we assume (a, b, c) => np.concatenate((a, b, c), axis), be careful when 
% deal the value of axis 
%
% INPUTS: 
% u2Trans, v2Trans - vectors with coordinates u and v of the transformed image point (p') 
% uBase, vBase - vectors with coordinates u and v of the original base image point p  
% 
% OUTPUT 
% H - a 3x3 Homography matrix  
% 
% Shreya Chawla, 7th May 2021 
'''
# Read left image and get 6 uv points
I_left = Image.open('Left.jpg')
plt.imshow(I_left)
uvBase = plt.ginput(6)  # Left image is taken as base
print('Left image uvBase  = ', uvBase)
uvBase = np.array(uvBase)
# Visualize points selected on left image
x = uvBase[:, 0]
y = uvBase[:, 1]
plt.scatter(x, y, marker='x', c="red", s=20, label='point')
plt.imshow(I_left)
plt.legend()
plt.title('Visualizing points selected on left image')
plt.show()

# Similarly get 6 corresponding uv points for right image
I_right = Image.open('Right.jpg')
plt.imshow(I_right)
uvTrans = plt.ginput(6)  # Right image is taken as Trans
print('Right image uvTrans = ', uvTrans)
uvTrans = np.array(uvTrans)
# Visualize points selected on right image
x = uvTrans[:, 0]
y = uvTrans[:, 1]
plt.scatter(x, y, marker='x', c="red", s=20, label='point')
plt.imshow(I_right)
plt.legend()
plt.title('Visualizing points selected on right image')
plt.show()

I_left = np.asarray(I_left)
I_right = np.asarray(I_right)

# Uncomment to normalize the uv coordinates
#uvBase, base_T = normalize_2d(I_left, uvBase)
#uvTrans, trans_T = normalize_2d(I_right, uvTrans)

# Split uvBase and uvTrans
uTrans, vTrans, uBase, vBase = uvTrans[:, 0], uvTrans[:, 1], uvBase[:, 0], uvBase[:, 1]

# Now get the 3x3 homography matrix H
H = homography(uTrans, vTrans, uBase, vBase)

# Uncomment to de-normalize H if normalized before
#H = np.dot(np.dot(trans_T, H), base_T)
# Display H
print('H = ', H)

# Warp left image according to H using cv2 inbuilt function
I_warp = cv2.warpPerspective(I_left, H, (I_left.shape[1], I_left.shape[0]))

# Display the warped image
f, axarr = plt.subplots(1, 2)
axarr[0].imshow(I_left)
axarr[0].title.set_text('Original left image')
axarr[1].imshow(I_warp)
axarr[1].title.set_text('Using my H matrix')
plt.show()


############################################################################
def rq(A):
    # RQ factorisation

    [q, r] = np.linalg.qr(A.T)  # numpy has QR decomposition, here we can do it
    # with Q: orthonormal and R: upper triangle. Apply QR
    # for the A-transpose, then A = (qr).T = r.T@q.T = RQ
    R = r.T
    Q = q.T
    return R, Q
