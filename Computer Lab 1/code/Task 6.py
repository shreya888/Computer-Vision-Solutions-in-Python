import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# Load the image2.jpg
dir_add = os.path.dirname(__file__)
image_add = os.path.abspath(os.path.join(dir_add, "..", "images/images", "image1.jpg"))
im = cv2.imread(image_add)
# Convert BGR to RGB for displaying later
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# Resizing the image to 512 x 512 - col x row
im_scaled = cv2.resize(im, (512, 512))
# Clipping the new image to have a minimum value of 0 and a maximum value of 255
im_scaled = np.clip(im_scaled, 0, 255)

# Display original image with the new cropped and scaled images for comparison
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.set_title('Original image')
ax1.imshow(im)
ax2 = fig.add_subplot(1,2,2)
ax2.set_title('Resized image')
ax2.imshow(im_scaled)
plt.show()

'''
my_translation function translates (shifts) input image by given pixels
    Args:
        im          (ndarray) RGB input image
        tx          (float) Amount shift in positive x direction
        ty          (float) Amount shift in positive y direction
    Displays:
        im_shifted  (ndarray) displys the translated image
'''
def my_translation(im, tx, ty):
    '''
    M = np.float32([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
    '''
    height, width = im.shape[:-1]

    tx = round(tx)
    ty = round(ty)

    # shift image by tx and ty
    im_shifted = np.roll(im, tx, axis=0)
    im_shifted = np.roll(im_shifted, ty, axis=1)
    im_shifted = cv2.rectangle(im_shifted, (0, 0), (width, tx), 0, -1)
    im_shifted = cv2.rectangle(im_shifted, (0, 0), (ty, height), 0, -1)

    # black out the shifted parts in image
    if ty < 0:
        im_shifted[:, ty:] = 0
    else:
        im_shifted[:, :ty] = 0
    if tx < 0:
        im_shifted[tx:, :] = 0
    else:
        im_shifted[:tx, :] = 0

    # Display the translated image
    plt.imshow(im_shifted)
    plt.title("Translated image by tx = "+str(tx)+", ty = "+str(ty))
    plt.xticks([]), plt.yticks([])
    plt.show()


my_translation(im_scaled, 2.0, 4.0)
my_translation(im_scaled, -4.0, -6.0)
my_translation(im_scaled, 2.5, 4.5)
my_translation(im_scaled, -0.9, 1.7)
my_translation(im_scaled, 92.0, -91.0)