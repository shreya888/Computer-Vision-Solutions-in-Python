import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# Load the gray scale image
dir_add = os.path.dirname(__file__)
image_add = os.path.abspath(os.path.join(dir_add, '..', 'images/images', 'Atowergray.jpg'))
im = cv2.imread(image_add)

# Negative image by subtracting the img from max value (found max value by finding np.iinfo on im's datatype)
im_neg = np.iinfo(im.dtype).max - im

# Show the original image and its negative side-by-side
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
ax[0].set_title('Original Image')
ax[0].imshow(im)
ax[1].set_title('Negative Image')
ax[1].imshow(im_neg)
# Remove x and y ticks
plt.xticks([])
plt.yticks([])
plt.show()

# Flip image horizontally
im_flip = cv2.flip(im, 1)

# Show the original image and its flipped side-by-side
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
ax[0].imshow(im)
ax[0].set_title('Original Image')
ax[1].imshow(im_flip)
ax[1].set_title('Horizontally Flipped Image')
# Remove x and y ticks
plt.xticks([])
plt.yticks([])
plt.show()

# Read a colour image
image_add2 = os.path.abspath(os.path.join(dir_add, '..', 'images/images', 'image1.jpg'))
im2 = cv2.imread(image_add2)

# Convert BGR to RGB
im_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

# Get the red and green colour channels using cv2.split function
R, G, _ = cv2.split(im_rgb)

# Swap the red and green colour channels of the input
im_swap = im_rgb.copy()
im_swap[:,:,1] = R
im_swap[:,:,0] = G

# Show the original image and swapped image side-by-side
fig, ax = plt.subplots(2)
# Remove x and y ticks
for i in range(2):
    ax[i].set_xticks([]), ax[i].set_yticks([])
ax[0].set_title('Original Image')
ax[0].imshow(im_rgb)
ax[1].set_title('Channel Swapped Image')
ax[1].imshow(im_swap)
plt.show()

# Average images of 2.1 and 2.2 and typecast it into int
im_avg = ((im + im_flip)/2).astype(int)
plt.imshow(im_avg)
plt.title('Average of original and horizontally flipped images')
# Remove x and y ticks
plt.xticks([])
plt.yticks([])
plt.show()

# Adding random integer as noise to grayscale image
row, col, ch = im.shape
noise = np.random.randint(0,127,(row, col, ch))
noise = noise.reshape(row, col, ch)
im_noisy = im + noise
# Clipping the new image to have a minimum value of 0 and a maximum value of 255 using numpy function
im_noisy = np.clip(im_noisy, 0, 255, out=im_noisy)
# Show the noisy image
plt.imshow(im_noisy)
plt.title('Noisy image')
# Remove x and y ticks
plt.xticks([])
plt.yticks([])
plt.show()