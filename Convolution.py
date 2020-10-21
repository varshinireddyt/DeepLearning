"""
Convolution using numpy

"""
import numpy as np
def conv_2d(image,kernel, bias):
    kernel_size = kernel.shape[0]
    image_size = image.shape[0]
    output_shape = image_size - kernel_size + 1  # output shape
    output = np.zeros((output_shape,output_shape))
    for row in range(image_size - 1):
        for col in range(image.shape[1]-1):
            window = image[row: row + kernel_size, col: col + kernel_size]
            output[row,col] = np.sum(np.multiply(kernel,window))
    return output + bias

image = np.array([[3., 9., 0.],
       [2., 8., 1.],
       [1., 4., 8.]])
kernel = np.array([[8., 9.],
       [4., 4.]])
bias = 0

print(conv_2d(image,kernel,bias))
