import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# Load the Figure 2 a and b images and convert them to RGB images
dir_add = os.path.dirname(__file__)
image_add_a = os.path.abspath(os.path.join(dir_add, '..', 'images/images', 'Figure2-a.png'))
im_a = cv2.imread(image_add_a)
im_a = cv2.cvtColor(im_a, cv2.COLOR_BGR2RGB)
image_add_b = os.path.abspath(os.path.join(dir_add, '..', 'images/images', 'Figure2-b.png'))
im_b = cv2.imread(image_add_b)
im_b = cv2.cvtColor(im_b, cv2.COLOR_BGR2RGB)


'''
Function to convert the RGB image to YUV colour space
    Args:
        im          (ndarray) RGB input image
    Returns:
        im_conv     (ndarray) output Y,U,V color channels of image
'''
def myRGB2YUV(im):
    # Defining the constants
    W_R = 0.299
    W_B = 0.114
    W_G = 1 - W_B - W_R
    U_max = 0.436
    V_max = 0.615
    # Get RGB color channels
    R, G, B = cv2.split(im)
    # Using the formula, convert into YUV
    Y = W_R * R + W_G * G + W_B * B
    U = U_max * (B - Y) / (1 - W_B)
    V = V_max * (R - Y) / (1 - W_R)
    return Y, U, V


# Convert Fig.2(a) with cvRGB2YUV function written above
channels = myRGB2YUV(im_a)
# For viewing separate Y,U,V channels
channel_name = ('Y', 'U', 'V')
# Display the YUV channels generated using cvRGB2YUV function
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
for i, ch in enumerate(channel_name):
    ax[i].imshow(channels[i])
    ax[i].set_title(ch + ' Channel')
# Remove x and y ticks
plt.xticks([])
plt.yticks([])
plt.show()

# Convert RGB image to YUV and separate them using inbuilt cv2 cvtColor function
im_yuv = cv2.cvtColor(im_a, cv2.COLOR_RGB2YUV)
Y_cv = im_yuv[:, :, 0]
U_cv = im_yuv[:, :, 1]
V_cv = im_yuv[:, :, 2]
channels_cv = (Y_cv, U_cv, V_cv)
# Display the YUV channels of image produced by cv2
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
for i, ch in enumerate(channel_name):
    ax[i].imshow(channels_cv[i])
    ax[i].set_title(ch + ' Channel (cv2)')
# Remove x and y ticks
plt.xticks([])
plt.yticks([])
plt.show()

# Using Canny edge detector to get regions in the image with 1 pixel thick edges
im_edges = cv2.Canny(im_b, cv2.CV_64F, 100, 200)

# Uncomment to display the edges 
'''
plt.imshow(im_edges, cmap="gray")
plt.show()
'''


'''
Helper function to find average intensity (y) given y channel of image
    Args:
        im_y        (ndarray) the y intensity channel of an image
    Prints:
        avg_y       (ndarray) prints the computer average intensity for each of the 5 regions in image
'''
def find_avg_y(im_y):
    # Stores the average y values
    avg_y = []
    # Loop over the image's row to get average y values
    for i in range(im_edges.shape[1]):
        # Check if pixel represents edge
        if im_edges.item(0, i) != 0:
            # If it is edge then the next pixel in row is not edge and part of region
            # As each region will have same intensity hence taking only 1 value is same as average intensity
            avg_y.append(im_y[0, i + 1])
    # Print the intensities
    print(avg_y)


# Using my RGB to YUV function defined above to find the average Y values
im_y, _, _ = myRGB2YUV(im_b)
find_avg_y(im_y)
# Using cv2 find the average Y values to compare with my values
im_yuv = cv2.cvtColor(im_b, cv2.COLOR_RGB2YUV)
im_y = im_yuv[:, :, 0]
find_avg_y(im_y)
