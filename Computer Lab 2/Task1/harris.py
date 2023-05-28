"""
Task 1: Harris Corner Detector
Author: Shreya Chawla (u7195872)
"""

# Import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt


def conv2(img, conv_filter):
    """
    Function to convolve image with a filter
    :param img: (numpy array) image to be convolved
    :param conv_filter: (numpy array) filter
    :return: (numpy array) resultant convolved image
    """
    # Flip the filter
    f_siz_1, f_size_2 = conv_filter.shape
    conv_filter = conv_filter[range(f_siz_1 - 1, -1, -1), :][:, range(f_siz_1 - 1, -1, -1)]
    pad = (conv_filter.shape[0] - 1) // 2
    result = np.zeros((img.shape))
    img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    filter_size = conv_filter.shape[0]
    for r in np.arange(img.shape[0] - filter_size + 1):
        for c in np.arange(img.shape[1] - filter_size + 1):
            curr_region = img[r:r + filter_size, c:c + filter_size]
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.
    return result


def fspecial(shape=(3, 3), sigma=0.5):
    """
    Function to generate a 2D Gaussian filter
    :param shape: (tuple) shape of window; default = (3,3)
    :param sigma: (int) Gaussian filter's standard deviation; default = 0.5
    :return: (numpy array) returns the generated Gaussian filter
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m: m+1, -n: n+1]  # Creates a multidimensional mesh-grid
    h = np.exp(-(x*x + y*y) / (2. * sigma * sigma))  # Fourier transform of x
    # finfo() gives machine limits for floating point types, eps is the difference between 1.0 and the next smallest
    # representable float larger than 1.0. Thus np.finfo(h.dtype).eps would be the smallest float in magnitude.
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()  # Sum of h array (which are > 0 in previous step)
    if sumh != 0:
        h /= sumh  # Normalize h if sum > 0
    return h


def my_harris(img, sigma, thresh, k):
    """
    Function to detect Haris corners on the given image
    :param img: (numpy array) image to be convolved
    :param sigma: (int) standard deviation to generate kernel/filter
    :param thresh: (float) threshold value
    :param k: (float) constant in cornerness formula
    :return: (list) list of all corners detected
    """
    # Find x and y gradient
    dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    dy = dx.transpose()  # 'dy' is transpose of 'dx'
    Ix = conv2(img, dx)
    Iy = conv2(img, dy)

    # Find corners in image
    # 'g' is the window function which will be used to find R (response). It is a Gaussian window function.
    # Dimensions of 'g' is neighborhood to be considered.
    g = fspecial((max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma)
    Iy2 = conv2(np.power(Iy, 2), g)  # y and y
    Ix2 = conv2(np.power(Ix, 2), g)  # x and x
    Ixy = conv2(Ix * Iy, g)  # x and y

    # Find if interest point using cornerness
    # Determinant
    det = Ix2 * Iy2 - Ixy * Ixy
    # Trace
    trace = Ix2 + Iy2
    # R is the cornerness or response
    R = det - k * trace * trace

    # Threshold based on max response value
    thresh *= np.amax(R)  # Threshold for an optimal value, it may vary depending on the image.

    # List of corners detected
    corners = []

    # If resonse > threshold then it is a corner else skip
    for row, response in enumerate(R):
        for col, r in enumerate(response):
            if r > thresh:
                corners.append([row, col, r])
    print('\nNumber of corners detected by my Harris corner detector = ', len(corners))
    return corners


def non_maximum_suppression(corners, dist):
    """
    Function performs non-maximum suppression of corners within a certain distance
    :param corners: (list) list of all initial corners detected
    :param dist: (int) pixel distance in which suppressed
    :return: (list) resultant corners after non-maximum suppression
    """
    # If no corners found then return that empty list
    if len(corners) == 0:
        return corners

    # Sort corners based on their cornerness value
    corners = sorted(corners, key=lambda c: c[2], reverse=True)
    # List of corners that are not suppressed
    chosen_corners = list()
    chosen_corners.append(corners[0][: -1])

    #  Compare corners to see if there are more corners in vicinity of the current corner
    # If not in the neighborhood of dist distance then that corner is chosen
    for corner in corners:
        for chosen in chosen_corners:
            if abs(corner[0] - chosen[0]) < dist and abs(corner[1] - chosen[1]) < dist:
                break
        else:
            chosen_corners.append(corner[: -1])
    print('Number of corners detected after non maximum suppression = ', len(chosen_corners))
    return chosen_corners


def plot_corner_img(img, title, corners):
    """
    Function to plot corners on the given image
    :param img: (numpy array) image to be displayed
    :param title: (string) title of plot
    :param corners: (list) list of corners
    :return:
    """
    x, y = [], []  # Store x and y coordinates of corners
    # Dealing with cases where no corners are detected
    if len(corners) == 0:
        plt.imshow(im, cmap='gray')
        plt.title('No corners detected ' + title)
        plt.show()
        return
    for corner in corners:
        row, col = corner[0], corner[1]
        x.append(row)
        y.append(col)
    # Plot corners image marked by crosses
    # if grayscale image then cmap will show gray image else colored
    plt.imshow(img, cmap='gray')
    plt.title('Corners detected: ' + title)
    plt.plot(y, x, 'x', color='red')
    plt.show()


# Parameters, add more if needed
sigma = 2
thresh = 0.01
k = 0.04  # Harris detector free parameter in the equation

# Store all image names in all_image_names
all_image_names = ['Harris-'+str(i)+'.jpg' for i in range(1, 7)]
distance = [10, 20, 10, 10, 50, 10]
# Now find corners for all images
for i, image_name in enumerate(all_image_names):
    # Read each image in im
    im = plt.imread(image_name)

    # If image is in RGB, convert to gray scale
    if 3 in im.shape:
        im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    else:
        im_gray = im

    # Find harris corners in image
    corners = my_harris(im_gray, sigma, thresh, k)

    # Plot the corners from my_harris
    plot_corner_img(im, image_name+' (result of my_harris)', corners)

    # Suppress corners using non_maximum_suppression at pixel distance = dis
    corners = non_maximum_suppression(corners, distance[i])

    # Plot the corners after non maximum suppression
    plot_corner_img(im, image_name+' (after suppression)', corners)

    # Now, find corners identified by cv2's inbuilt harris corner detector
    print('Inbuilt cv2 cornerHarris function')
    im_gray = np.float32(im_gray)
    dst = cv2.cornerHarris(im_gray, int(max(1, np.floor(3 * sigma) * 2 + 1)), 3, k)
    # Show the inbuilt function's corners
    corners = []
    for x in range(dst.shape[0]):
        for y in range(dst.shape[1]):
            if dst[x, y] > thresh * np.amax(dst):  # Threshold for an optimal value, it may vary depending on the image
                corners.append(np.array([x, y, dst[x, y]]))
    # Now perform non-maximum suppression
    corners = non_maximum_suppression(corners, distance[i])
    # Plot these corners obtained from in-built function
    plot_corner_img(im, image_name + ' (inbuilt function)', corners)
