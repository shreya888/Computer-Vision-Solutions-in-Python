# Import numpy
import numpy as np
# 1
a = np.array([[2, 3, 4], [5, 2, 200]])  # Creates new np array of dimension (2,3)
print(a)
# 2
b = a[:, 1]  # Returns elements of column of index 1 (i.e. 2nd column) of array a as an array stored in b
print(b)
# 3
f = np.random.randn(400,1)+3  # Returns a floating point array shaped (400,1) of random values and each element of array is incremented by 3
# Basically a Gaussian distribution with mu = 3 and sigma = 1 in a column array
print(f)
# 4
g = f[ f > 0 ]*3  # For all values in f array if they are greater than 0 then multiplied with 3 and stores only the multiplied values in 1D array g
print(g)
# 5
x = np.zeros(100) + 0.45  # Initializes x array with (100,) shaped 1D array with value of 0.45 each
print(x)
# 6
y = 0.5 * np.ones([1, len(x)])  # Initializes y array with 100 values (from len(x)) of 0.5 each of dimension (1,100) i.e. a 2D array with only 1 row of 100 values
print(y)
# 7
z = x + y  # Adds array x and y resulting in an array of dimension (1,100) each having value 0.95 (as 0.45 + 0.5 = 0.95)
print(z)
# 8
a = np.linspace(1, 499, 250, dtype=int)  # Returns 250 evenly spaced numbers in an np array over the specified interval of 1 to 499 (inclusive by default) of type int
print(a)
# 9
b = a[::-2]  # Starting from the end of array a all elements are taken at a step of 2 (skipping one value) and the resulting array is stored in b
print(b)
# 10
b[b > 50] = 0  # Replaces elements of b array which have value greater than 50 by 0 and stores this array in b
print(b)