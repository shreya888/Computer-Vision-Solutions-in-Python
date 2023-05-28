import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

'''
my_kmeans function: Performs K-means algorithm on data and returns the K clusters
'''
def my_kmeans(data, K):
    # Step 1 Initialize the K centroids randomly
    centroids = np.random.random((K, data.shape[1]))
    print('Initialized centroids to: ', centroids)
    start_time = time.time()  # Keep track of convergence time
    clusters = np.zeros_like(data)  # Keeps track of which datapoint belongs to which cluster
    diff = True  # Boolean which keeps track of when to stop - reached local maximum

    while diff:
        new_centroids = np.zeros_like(centroids)  # New centroids calculated are stored in this array

        # Step 2 Find Euclidean distance (L2 norm) of each point from each centroid and store it in a dictionary
        for i in range(data.shape[0]):
            dist = dict()  # Dictionary to store distance of each centroid from each point
            for j in range(K):
                dist.update({j: np.linalg.norm(data[i] - centroids[j])})
            # Step 3 Assign points to closest cluster (or centroid)
            index = [key for key in dist if dist[key] == min(dist.values())]
            if len(index) > 1:  # If more than 1 centroids are at minimum distance
                index = index[0]
            clusters[i] = centroids[index]

        # Step 4 Compute the new centroid
        for i in range(K):
            cluster = np.where((clusters == centroids[i]).all(axis=1))[0]
            if cluster.shape[0] > 0:
                new_centroids[i] = sum(data[cluster]) / cluster.shape[0]
            else:
                new_centroids[i] = centroids[i]

        # Step 5 If no difference between between new and old set of centroids, then stop, else repeat step 2 to 4
        if np.isclose(centroids, new_centroids, 1e-15).all():
            diff = False
        else:
            centroids = new_centroids
    end_time = time.time()
    print('Time taken for the k means algorithm to converge = ', end_time - start_time)
    return clusters


'''
my_kmeans_pp function: Performs K-means++ algorithm on data and returns the K clusters
'''
def my_kmeans_pp(data, K):
    # Step 1 Initialize the K centroids randomly
    centroids = []
    # 1.1) Choose the 1st centroid uniformly sampling from the data points in data
    centroids.append(data[np.random.randint(data.shape[0])])
    # 1.2) For the next K-1 centroids, we need to choose them as far from the 1st as possible
    for c_id in range(K - 1):
        dist = []  # Shortest distance between point and closest centroid (D(x))
        for i in range(data.shape[0]):
            point = data[i, :]
            # For every point in X, choose the min distance between
            # np.inf and L2 norm (or euclidean distance between points and the centroids)
            d = np.inf
            for j in range(len(centroids)):
                temp_dist = np.linalg.norm(point - centroids[j])
                d = min(d, temp_dist)
            dist.append(d)

        # Select data point with maximum distance from the 1st centroid as the next centroid
        dist = np.array(dist)
        next_centroid = data[np.argmax(dist)]
        centroids.append(next_centroid)
        # 1.3) Repeat 1.1 and 1.2 until there are K clusters

    centroids = np.array(centroids)

    print('Initialized centroids to: \n', centroids)
    start_time = time.time()  # Keep track of convergence time
    clusters = np.zeros_like(data)  # Keeps track of which datapoint belongs to which cluster
    diff = True  # Boolean which keeps track of when to stop - reached local maximum

    while diff:
        new_centroids = np.zeros_like(centroids)  # New centroids calculated are stored in this array
        # Step 2 Find Euclidean distance (L2 norm) of each point from each centroid and store it in a dictionary
        for i in range(data.shape[0]):
            dist = dict()  # Dictionary to store distance of each centroid from each point
            for j in range(K):
                dist.update({j: np.linalg.norm(data[i] - centroids[j])})
            # Step 3 Assign points to closest cluster (or centroid)
            index = [key for key in dist if dist[key] == min(dist.values())]
            if len(index) > 1:  # If more than 1 centroids are at minimum distance
                index = index[1]
            clusters[i] = centroids[index]

        # Step 4 Compute the new centroid
        for i in range(K):
            cluster = np.where((clusters == centroids[i]).all(axis=1))[0]
            if cluster.shape[0] > 0:
                new_centroids[i] = sum(np.abs(data[cluster])) / cluster.shape[0]
            else:
                new_centroids[i] = centroids[i]

        # Step 5 If no difference between between new and old set of centroids, then stop, else repeat step 2 to 4
        if (centroids - new_centroids).sum() == 0:
            diff = False
        else:
            centroids = new_centroids
    end_time = time.time()
    print('Time taken for the k-means++ algorithm to converge = ', end_time - start_time)
    return clusters


'''
normalize function: Normalizes input's column by dividing it by its L2 norm
'''
def normalize(input):
    output = input
    for i in range(input.shape[1]):
        output[:, i] = input[:, i] / np.linalg.norm(input[:, i])
    return output


# Read the images using image name
all_image_names = ['peppers.png', 'mandm.png']
for image_name in all_image_names:
    print('Segmentation on ',image_name)
    im = plt.imread(image_name)
    plt.imshow(im)
    plt.title(image_name)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    # Convert 16 bit image to 8 bit
    if im.dtype == np.uint16:
        im = im.astype(np.uint8)
    if im.dtype == np.float32:
        im *= 255  # or any coefficient
        im = im.astype(np.uint8)
    print(image_name, ' is of type: ', im.dtype)
    # Convert to LAB color space
    im = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)

    # Show the image in LAB space
    plt.imshow(im)
    plt.title('Image in LAB space')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # Number of clusters K
    K = 3

    with_coordinates = False  # Boolean for how the segmentation is to be carried out(with or without pixel coordinates)
    clusters_k = np.array([])  # Stores all the clusters returned from my_kmeans method (K-means)
    clusters_k_pp = np.array([])  # Stores all the clusters returned from my_kmeans_pp method (K-means++)
    if with_coordinates:  # With pixel coordinates
        # Create 5-D image data: L, A, B, X, Y
        fiveD_data = list()
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                fiveD_data.append(np.append(im[i, j], np.array([i, j])))
        fiveD_data = np.array(fiveD_data)
        fiveD_data = normalize(fiveD_data)
        clusters_k = my_kmeans(fiveD_data, K)
        # Take only first 3 values (L,A,B) to show image, drop x,y coordinate of centroid
        clusters_k = np.reshape(clusters_k[:, :3], im.shape)
        # Similarly for K-means++ clusters
        clusters_k_pp = my_kmeans_pp(fiveD_data, K)
        # Take only first 3 values (L,A,B) to show image, drop x,y coordinate of centroid
        clusters_k_pp = np.reshape(clusters_k_pp[:, :3], im.shape)

    else:  # Without pixel coordinates
        # Create 3-D image data: L, A, B
        threeD_data = np.reshape(im, (im.shape[0] * im.shape[1], 3))
        threeD_data = normalize(threeD_data)
        clusters_k = my_kmeans(threeD_data, K)
        clusters_k_pp = my_kmeans_pp(threeD_data, K)
        clusters_k = clusters_k.reshape(im.shape)
        clusters_k_pp = clusters_k_pp.reshape(im.shape)

    # Convert back to RGB color space
    plt.imshow(cv2.cvtColor(clusters_k, cv2.COLOR_LAB2RGB))
    plt.title('Segmented image (my_kmeans)')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # Convert back to RGB color space
    plt.imshow(cv2.cvtColor(clusters_k_pp, cv2.COLOR_LAB2RGB))
    plt.title('Segmented image (my_kmeans_pp)')
    plt.xticks([])
    plt.yticks([])
    plt.show()
