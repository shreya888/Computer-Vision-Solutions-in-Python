import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# All images are supposed to be the same size, say, N*L
dir_add = os.path.dirname(__file__)
TRAIN_DIR = os.path.abspath(os.path.join(dir_add, 'Yale-FaceA/trainingset'))
MY_DIR = os.path.abspath(os.path.join(dir_add, 'Yale-FaceA/my_images'))
#MY_TEST_DIR = os.path.abspath(os.path.join(dir_add, 'Yale-FaceA/my_images'))


def get_file_names(root):
    filenames = list()
    for root, _, files in os.walk(root):
        for f in files:
            filename = os.path.join(root, f)
            filenames.append(filename)
    return filenames

# Read all training image files
train_images = get_file_names(TRAIN_DIR)
my_images = get_file_names(MY_DIR)
numTrainingFaces = len(my_images)

train_im = plt.imread(train_images[0])  # For reference
h, w = train_im.shape
for image in my_images:
    im = cv2.imread(image)
    # Convert 16 bit image to 8 bit
    if im.dtype == np.uint16:
        im = im.astype(np.uint8)
        # print('Changed to uint8')
    if im.dtype == np.float32:
        im *= 255  # or any coefficient
        im = im.astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (w, h))
    # Save image
    cv2.imwrite(image, im)