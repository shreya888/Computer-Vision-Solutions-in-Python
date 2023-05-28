import os
import numpy as np
import matplotlib.pyplot as plt

# The directory address where all images are stored
DIR_ADD = os.path.dirname(__file__)
TRAIN_DIR = os.path.abspath(os.path.join(DIR_ADD, 'Yale-FaceA/trainingset'))
TEST_DIR = os.path.abspath(os.path.join(DIR_ADD, 'Yale-FaceA/testset'))

'''
get_file_names function: returns file names in root directory
'''
def get_file_names(root):
    filenames = list()
    for root, _, files in os.walk(root):
        for f in files:
            filename = os.path.join(root, f)
            filenames.append(filename)
    return filenames

'''
get_datamatrix function: Given matrix X, it reshapes it into a datamatrix with each column as an image
'''
def get_datamatrix(X):
    # Get image dimensions
    x, y = X[0].shape
    n = x * y
    # Define datamatrix with each image as column
    datamatrix = np.array(np.zeros((n, len(X))))
    for i, image in enumerate(X):
        datamatrix[:, i] = np.reshape(image, (n, ))

    return datamatrix


'''
pca function: Performs pca on matrix X and returns top k Eigenvalues, Eigenvectors, and mean of matrix X
'''
def pca(X, k):
    mean = X.mean(axis=0)
    # Subtract mean from images or datamatrix X
    X = X - mean

    n, m = X.shape
    if n > m:  # Use transpose trick when one of the dimensions of matrix is greater than other
        C = X.T @ X  # Covariance Matrix
        eigval, eigvec = np.linalg.eigh(C)
        eigvec = X @ eigvec
        for i in range(m):
            eigvec[:, i] = eigvec[:, i] / np.linalg.norm(eigvec[:, i])
    else:
        C = X @ X.T  # Covariance Matrix
        eigval, eigvec = np.linalg.eigh(C)

    # Sort eigenvectors descending by their eigenvalue
    idx = np.argsort(-eigval)
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]
    # select only top k
    eigval = eigval[0: k].copy()
    eigvec = eigvec[:, 0: k].copy()

    return eigval, eigvec, mean


'''
plot_top_k function: Plots top k Eigenfaces
'''
def plot_top_k(title, images, rows, cols, figsize=None):
    if figsize == None:
        fig = plt.figure(figsize=(rows*3, cols*3))
    else:
        fig = plt.figure(figsize=figsize)
    # Figure title
    fig.text(.5, .9, title, horizontalalignment='center')
    for i in range(len(images)):
        ax0 = fig.add_subplot(rows, cols, (i + 1))
        plt.setp(ax0.get_xticklabels([]), visible=False)
        plt.setp(ax0.get_yticklabels([]), visible=False)
        plt.title(i + 1)
        plt.imshow(np.asarray(images[i]), cmap='gray')
    plt.show()


'''
project function: Projects normalized X onto U (eigenvector)
'''
def project (U, X, mean):
    return U.T @ (X - mean)



# Read all training image files
train_images = get_file_names(TRAIN_DIR)
numTrainingFaces = len(train_images)

train_faces = list()

for image in train_images:
    train_faces.append(plt.imread(image))

# Calculate average face
mean_face = np.mean(train_faces, axis=0)

# Display mean_face
plt.imshow(mean_face, cmap='gray')
plt.title('Mean Face')
plt.xticks([])
plt.yticks([])
plt.show()

# Reshape the faces array such that each image is a column after flattening it
datamatrix = get_datamatrix(train_faces)

k = 10  # Parameter

# Calculate PCA with k=10
eigval, eigvec, mean = pca(datamatrix, k)

# Convert top k eigenfaces from column vectors to images
im_list = []
for i in range(eigvec.shape[1]):
    evec = eigvec[:, i].reshape(train_faces[0].shape)  # Convert ith eigenvector
    im_list.append(np.asarray(evec))
# Plot them the top k eigenfaces
plot_top_k(title="Top "+str(k)+" Eigenfaces", images=im_list, rows=3, cols=4)

# Now read test the 10 test images in the Yale-Face dataset
test_images = get_file_names(TEST_DIR)
numTestingFaces = len(test_images)

# Run recognition for each test image
# Step 1: Project all training images onto subspace spanned by principal components (get weights)
train_proj = project(eigvec, datamatrix, mean)  # Store projections for every image
# Step 2: Project the test image onto PCA subspace one by one
for i, image in enumerate(test_images):
    # Read test images one at a time
    im = plt.imread(image)
    # First flattening image
    im_flat = im.reshape(-1, 1)
    # Project the normalized image vector
    test_proj = project(eigvec, im_flat, mean)
    min_error = [np.inf]*3  # Minimum error found till now (used to find closest image)
    min_image_idx = [10]*3  # Index of image which is closest to test
    # Find reconstruction error
    for j in range(len(train_proj)):
        error = np.linalg.norm(test_proj - train_proj[j])
        # Now find the closest training face in k-dimensional space by minimizing reconstruction error
        if error < min_error[2]:
            min_error[2] = error
            min_image_idx[2] = j
        if error < min_error[1]:
            min_error[1] = error
            min_image_idx[1] = j
        if error < min_error[0]:
            min_error[0] = error
            min_image_idx[0] = j  # The image index which has least error
    # Plot the results
    plot_top_k(title='First 3 faces predicted for subject: '+str(i+1),
               images=[im, train_faces[min_image_idx[0]], train_faces[min_image_idx[1]], train_faces[min_image_idx[2]]],
               rows=1, cols=4, figsize=(10, 5))

#######################################################################################################################

# Repeat above for my test image
# Now lets read test image from my_image directory's test folder
MY_DIR_TEST = os.path.abspath(os.path.join(DIR_ADD, 'Yale-FaceA/my_images/test'))
my_test_images = get_file_names(MY_DIR_TEST)
for image in my_test_images:
    im = plt.imread(image)
    # First flattening image
    im_flat = im.reshape(-1, 1)
    # Project the normalized image vector
    test_proj = project(eigvec, im_flat, mean)
    min_error = [np.inf] * 3  # Minimum error found till now (used to find closest image)
    min_image_idx = [10] * 3  # Index of image which is closest to test
    # Find reconstruction error
    for j in range(len(train_proj)):
        error = np.linalg.norm(test_proj - train_proj[j])
        # Now find the closest training face in k-dimensional space by minimizing reconstruction error
        if error < min_error[2]:
            min_error[2] = error
            min_image_idx[2] = j
        if error < min_error[1]:
            min_error[1] = error
            min_image_idx[1] = j
        if error < min_error[0]:
            min_error[0] = error
            min_image_idx[0] = j  # The image index which has least error
    # Plot the results
    plot_top_k(title='First 3 faces predicted for my test image',
               images=[im, train_faces[min_image_idx[0]], train_faces[min_image_idx[1]], train_faces[min_image_idx[2]]],
               rows=1, cols=4, figsize=(10, 5))

########################################################################################################################
# Read all of my images for training
MY_DIR_TRAIN = os.path.abspath(os.path.join(DIR_ADD, 'Yale-FaceA/my_images/train'))
my_train_images = get_file_names(MY_DIR_TRAIN)
# Append to already existing trainset
for image in my_train_images:
    train_faces.append(plt.imread(image))

# Reshape the faces array such that each image is a column after flattening it
datamatrix = get_datamatrix(train_faces)

# Find principal components again
datamatrix = get_datamatrix(train_faces)
eigval, eigvec, mean = pca(datamatrix, k)

# Find the projection of new train set
train_proj = project(eigvec, datamatrix, mean)

for image in my_test_images:
    im = plt.imread(image)
    # First flattening image
    im_flat = im.reshape(-1, 1)
    # Project the normalized image vector
    test_proj = project(eigvec, im_flat, mean)
    min_error = [np.inf] * 3  # Minimum error found till now (used to find closest image)
    min_image_idx = [10] * 3  # Index of image which is closest to test
    # Find reconstruction error
    for j in range(len(train_proj)):
        error = np.linalg.norm(test_proj - train_proj[j])
        # Now find the closest training face in k-dimensional space by minimizing reconstruction error
        if error < min_error[2]:
            min_error[2] = error
            min_image_idx[2] = j
        if error < min_error[1]:
            min_error[1] = error
            min_image_idx[1] = j
        if error < min_error[0]:
            min_error[0] = error
            min_image_idx[0] = j  # The image index which has least error
    # Plot the results
    plot_top_k(title='First 3 faces predicted for my test image after training on my images as well',
               images=[im, train_faces[min_image_idx[0]], train_faces[min_image_idx[1]], train_faces[min_image_idx[2]]],
               rows=1, cols=4, figsize=(10, 5))