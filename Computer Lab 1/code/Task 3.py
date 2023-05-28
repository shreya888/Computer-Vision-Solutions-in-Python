import cv2
import os
import matplotlib.pyplot as plt

# Loading color image
dir_add = os.path.dirname(__file__)
image_add = os.path.abspath(os.path.join(dir_add, "..", "images/images", "image1.jpg"))
im = cv2.imread(image_add)

# Resizing image to 384 x 256 - col x row
im_scaled = cv2.resize(im, (384, 256))
# Convert BGR to RGB for im_scaled and show scaled image
im_scaled_rgb = cv2.cvtColor(im_scaled, cv2.COLOR_BGR2RGB)
plt.imshow(im_scaled_rgb)
plt.title('Scaled image (384 x 256)')
plt.xticks([])
plt.yticks([])
plt.show()

# For viewing separate Color Channels (in BGR order for cv2)
color = ('Blue', 'Green', 'Red')

# OpenCV's 'split' function splits the image into each color index (cv2 has the color order of BGR)
channels = cv2.split(im)

# Display each RGB channel
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
for i, col in enumerate(color):
    ax[i].imshow(channels[i],cmap="gray")
    ax[i].set_title(col+' Channel')
# Remove x and y ticks
plt.xticks([])
plt.yticks([])
plt.show()

#  Plot the histograms for each color channel
fig, ax = plt.subplots(3, sharex=True, sharey=True)
for i, col in enumerate(color):
    histogram = cv2.calcHist([im], [i], None, [256], [0, 256])
    ax[i].plot(histogram, color=col)
    ax[i].set_title(col+' Channel')
    plt.xlim([0, 256])
plt.show()

# Apply histogram equalisation to the resized image's grayscale image and its three channels separately
eq = []  # Contains the equalized image for all channels
eq_hist = []  # Contains equalized histogram for each channel
fig, ax = plt.subplots(3, sharex=True, sharey=True)
for i, col in enumerate(color):
    eq.append(cv2.equalizeHist(channels[i]))
    eq_hist.append(cv2.calcHist([eq[i]], None, None, [256], [0, 256]))
    ax[i].plot(eq_hist[i], color=col)
    ax[i].set_title('Equalized ' + col+' Channel')
    plt.xlim([0, 256])
plt.show()

# Display the equalized BGR channels of image
fig, ax = plt.subplots(3, sharex=True, sharey=True)
for i, col in enumerate(color):
    ax[i].imshow(eq[i],cmap="gray")
    ax[i].set_title('Equalized ' + col + ' Image')
plt.xticks([])
plt.yticks([])
plt.show()

# Display the equalized image intensity and image after equalization of intensity
im_yuv = cv2.cvtColor(im_scaled, cv2.COLOR_BGR2YUV)
Y = im_yuv[:,:,0]
eq = cv2.equalizeHist(Y)
eq_hist = cv2.calcHist([eq], None, None, [256], [0, 256])
fig, ax = plt.subplots(2)
ax[0].plot(eq_hist)
ax[0].set_title('Equalized Image Intensities')
ax[1].imshow(eq,cmap="gray")
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title('Equalized Image')
plt.show()
