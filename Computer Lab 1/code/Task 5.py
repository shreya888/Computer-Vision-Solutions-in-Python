import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# Load the image2.jpg
dir_add = os.path.dirname(__file__)
image_add = os.path.abspath(os.path.join(dir_add, "..", "images/images", "image2.jpg"))
im = cv2.imread(image_add)

# Cropping image corresponding to the central part
# min_dim corresponds to the minimum(height, weight)
min_dim = min(im.shape[0], im.shape[1])

# diff stores the difference between min(height, width) and the height and width
diff = []
for i in range(0, 2):
    diff.append(im.shape[i] - min_dim)

# Crop a square image
im_crop = im[diff[0] // 2: diff[0] // 2 + min_dim, diff[1] // 2: diff[1] // 2 + min_dim, :]

# Resizing the square image to 512 x 512 - col x row
im_scaled = cv2.resize(im_crop, (512, 512))
# Clipping the new image to have a minimum value of 0 and a maximum value of 255
im_scaled = np.clip(im_scaled, 0, 255, out=im_scaled)

# Convert the original image to RGB
im_RGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# Convert the new scaled image to grayscale image
im_scaled_gray = cv2.cvtColor(im_scaled, cv2.COLOR_BGR2GRAY)

# Add Gaussian noise to im_scaled with mean = 0 and standard deviation (sd) = 15
row, col = im_scaled_gray.shape
mean = 0
sd = 15
gauss = sd * np.random.randn(row, col) + mean
im_noisy = im_scaled_gray + gauss
# Clipping the noisy image to have a minimum value of 0 and a maximum value of 255
im_noisy = im_noisy.astype(np.uint8)

# Display original image, cropped, scaled and gray image for comparison
fig, ax = plt.subplots(2)
for i in range(2):
    ax[i].set_xticks([])
    ax[i].set_yticks([])
ax[0].set_title('Original image')
ax[1].set_title('Cropped, Resized and Grayscale image')
ax[0].imshow(im_RGB)
ax[1].imshow(im_scaled_gray, cmap="gray")
plt.show()

# Display noisy image
plt.title('Gaussian noise in image (mean=0, sd=15)')
plt.xticks([]), plt.yticks([])
plt.imshow(im_noisy, cmap="gray")
plt.show()

# Display histograms of before (im_scaled_gray) and after(im_noisy) adding the noise
fig, ax = plt.subplots(2)
hist = []
ax[0].set_title('Before noise')
ax[1].set_title('After noise')
# np.ravel returns a flattened 1-D array containing the elements of the input array
ax[0].hist(im_scaled_gray.ravel(), 256, [0, 256])
ax[1].hist(im_noisy.ravel(), 256, [0, 256])
plt.xlim([0, 256])
plt.show()

'''
Helper function for my_Gauss_filter and my_Bilateral_filter
    Args:
        im          (ndarray) monochrome input image
        kernel      (ndarray)   gaussian kernel to be convolved over image
    Returns:
        im_conv     (ndarray) output convolved image
'''
# Convolve image with kernel
def convolution(im, kernel):
    # Kernel dimensions (generalized for all types of kernels)
    k_h = kernel.shape[0]
    k_w = kernel.shape[1]

    # Padding image to be convolved using pad function of numpy
    # For grayscale image
    if len(im.shape) == 2:
        im_pad = np.pad(im, pad_width=((k_h//2, k_h//2), (k_w//2, k_w//2)),
                        mode='constant', constant_values=0).astype(np.float32)
    # For challenge task - color image
    elif len(im.shape) == 3:
        im_pad = np.pad(im, pad_width=((k_h//2, k_h//2), (k_w//2, k_w//2), (0, 0)),
                        mode='constant', constant_values=0).astype(np.float32)
    # If image has less than 2 channels then image is incorrect and the function is left
    else:
        print("Image incorrect")
        pass

    # Initializing convolved image
    im_conv = np.zeros_like(im_pad)

    h = k_h // 2
    w = k_w // 2

    # Using the formula from the lecture slides
    # Element wise matrix multiplication
    for i in range(h, im_pad.shape[0] - h):
        for j in range(w, im_pad.shape[1] - w):
            x = im_pad[i-h: i-h+k_h, j-w: j-w+k_w]
            im_conv[i][j] = np.sum(x.flatten() * kernel.flatten())
    if h == 0:
        return im_conv[h:, w: -w]
    if w == 0:
        return im_conv[h: -h, w:]
    return im_conv[h: -h, w: -w]


'''
Function to get smoothen image using Gaussian Filter
Performs gaussian filtering of an input image.
    Args:
        im               (ndarray) monochrome input image
        gaussKernel      (ndarray)   gaussian kernel to be convolved over image
    Returns:
        im_gauss         (ndarray) output gaussian-filtered image
'''
def my_Gauss_filter(im, gaussKernel):
    # Convolve image with gaussKernel to get a smoothened image im_gauss (convolution function defined above)
    im_gauss = convolution(im, gaussKernel)
    return im_gauss.astype(np.uint8)


# Generating Gaussian Kernels
sigma = 3
k_size = 5
gaussKernel = cv2.getGaussianKernel(k_size, sigma)

# Get my Gaussian filtering result
im_gauss = my_Gauss_filter(im_noisy, gaussKernel)
plt.xticks([]), plt.yticks([])
plt.title("Result of my_Gauss_filter (k_size=(5,5), sigma="+str(sigma))
plt.imshow(im_gauss, cmap="gray")
plt.show()

# Comparing different sigma
sigmas = [3, 7, 13]
fig, ax = plt.subplots(len(sigmas))
for i in range(len(sigmas)):
    gaussK = cv2.getGaussianKernel(k_size, sigmas[i])
    im_sigma = my_Gauss_filter(im_noisy, gaussK)
    ax[i].set_xticks([]), ax[i].set_yticks([])
    ax[i].set_title("my_Gauss_filter: sigma="+str(sigmas[i]))
    ax[i].imshow(im_sigma, cmap="gray")
plt.show()

# Compare my Gaussian filter result with inbuilt filter result
cv_gauss = cv2.GaussianBlur(im_noisy, (5,5), 3)
plt.title("Inbuilt Gaussian Filtering (k_size=(5,5), sigma=3)")
plt.xticks([]), plt.yticks([])
plt.imshow(cv_gauss, cmap="gray")
plt.show()

########################## CHALLENGE TASK ##########################
def my_Bilateral_filter(noisy, gaussKernel, colour_sigma):
    im_out = np.zeros_like(noisy)
    return im_out

# Generate noisy color image
gauss_col = sd * np.random.randn(row, col, 3) + mean
im_noisy_col = (im_scaled + gauss_col).astype(int)
im_noisy_col = np.clip(im_noisy_col, 0, 255)

plt.imshow(im_noisy_col)
plt.show()

'''def filter_bilateral( img_in, sigma_s, sigma_v):
    """Simple bilateral filtering of an input image
    Performs standard bilateral filtering of an input image. If padding is desired,
    img_in should be padded prior to calling
    Args:
        img_in       (ndarray) monochrome input image
        sigma_s      (float)   spatial gaussian std. dev.
        sigma_v      (float)   value gaussian std. dev.
    Returns:
        result       (ndarray) output bilateral-filtered image
    """
    # make a simple Gaussian function taking the squared radius
    gaussian = lambda r2, sigma: (np.exp( -0.5*r2/sigma**2 )*3).astype(int)*1.0/3.0

    # define the window width to be the 3 time the spatial std. dev. to
    # be sure that most of the spatial kernel is actually captured
    win_width = int( 3*sigma_s+1 )

    # initialize the results and sum of weights to very small values for
    # numerical stability. not strictly necessary but helpful to avoid
    # wild values with pathological choices of parameters
    wgt_sum = np.ones( img_in.shape )
    result  = img_in

    # accumulate the result by circularly shifting the image across the
    # window in the horizontal and vertical directions. within the inner
    # loop, calculate the two weights and accumulate the weight sum and
    # the unnormalized result image
    for shft_x in range(-win_width,win_width+1):
        for shft_y in range(-win_width,win_width+1):
            # compute the spatial weight
            w = gaussian( shft_x**2+shft_y**2, sigma_s )

            # shift by the offsets
            off = np.roll(img_in, [shft_y, shft_x], axis=[0,1] )

            # compute the value weight
            tw = w*gaussian( (off-img_in)**2, sigma_v )

            # accumulate the results
            result += off*tw
            wgt_sum += tw

    # normalize the result and return
    return result/wgt_sum


B = np.stack([
        filter_bilateral(im_noisy[:,:,0], 5, 5),
        filter_bilateral(im_noisy[:,:,1], 5, 5),
        filter_bilateral(im_noisy[:,:,2], 5, 5)], axis=2)

# stack the images horizontally
O = np.hstack( [im_noisy,B] )

# write out the image
plt.imshow(O*255.0 )
filter_bilateral(im_noisy, 5, 5)
'''