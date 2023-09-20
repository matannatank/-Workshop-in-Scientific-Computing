from numpy import array, exp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import time
import numpy as np
import math
import pyfftw.interfaces.numpy_fft as fft
import cmath
from scipy.fftpack import fft2, ifft2
from numpy.testing import assert_allclose
from PIL import Image
from sklearn.linear_model import LinearRegression
from scipy import ndimage
from skimage import io
from matplotlib import cm
from sklearn.metrics import mean_squared_error
from PIL import Image
import imageio
import os
from PIL import Image
import numpy as np
from scipy import fft
from scipy.fftpack import dct
import zlib
from scipy.fftpack import idct
import matplotlib.pyplot as plt
import cv2
import math
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
import os
from skimage.metrics import structural_similarity as ssim


# HW 1
# QU 1

# define dft forward algorithm
def dft_forward(vec):
    len_vec = len(vec)
    result = np.zeros(len_vec, dtype=np.complex128)
    for k in range(len_vec):
        for i in range(len_vec):
            result[k] += vec[i] * np.exp(2j * np.pi * i * k / len_vec)
    return result


# define dft backward algorithm
def dft_backward(vec):
    len_vec = len(vec)
    result = np.zeros(len_vec, dtype=complex)
    for i in range(len_vec):
        for k in range(len_vec):
            result[i] += vec[k] * np.exp(-2j * np.pi * i * k / len_vec)
        result[i] /= len_vec
    return result


# now we build function as the question which get vector and direction of dft and calculate
def dft_1d(vec, direction):
    if direction == 'FFTW_FORWARD':
        return dft_forward(vec)

    if direction == 'FFTW_BACKWARD':
        return dft_backward(vec)


# define vector for our tests \found that vector in the internet
vec = [-2.33486982e-16 + 1.14423775e-17j, 8.00000000e+00 - 1.25557246e-15j,
       2.33486982e-16 + 2.33486982e-16j, 0.00000000e+00 + 1.22464680e-16j,
       -1.14423775e-17 + 2.33486982e-16j, 0.00000000e+00 + 5.20784380e-16j,
       1.14423775e-17 + 1.14423775e-17j, 0.00000000e+00 + 1.22464680e-16j]


# QU 2

# build a matrix for dft forward
def dft_forward_matrix(n):
    # create  n*n matrix
    w = np.zeros((n, n), dtype=np.complex128)

    # Calculate the values of the matrix
    for i in range(n):
        for j in range(n):
            w[i, j] = np.exp(2j * np.pi * i * j / n)
    return w


# build a matrix for dft backward
def dft_backward_matrix(n):
    # create  n*n matrix
    w = np.zeros((n, n), dtype=np.complex128)

    # Calculate the values of the matrix
    for i in range(n):
        for j in range(n):
            w[i, j] = (1 / n) * np.exp(-2j * np.pi * i * j / n)
    return w


# build a function as the question which get vector and direction and build the matrix as required
# the function return matrix nxn
def dft_1d_matrix(vec, direction):
    len_vec = len(vec)
    vec = np.array(vec)

    if direction == 'FFTW_FORWARD':
        return vec.dot(dft_forward_matrix(len_vec))

    if direction == 'FFTW_BACKWARD':
        return vec.dot(dft_backward_matrix(len_vec))


# in this project we assume which n is a power of 2

# HW 2

"""# task 1"""

def is_power_of_two(num):
    if num <= 0:
        return False
    return (num & (num - 1)) == 0


def fft_1d_recursive_radix2(vec, direction):
    len_vec = len(vec)

    if is_power_of_two(len_vec) == False:
        return ValueError('the input isnt power of 2 ')

    if direction == 'FFTW_FORWARD':
        return fft_rec(vec)

    if direction == 'FFTW_BACKWARD':
        return ifft_rec(vec)


def fft_rec(x):
    # Base case , if the length of our input is one
    n = len(x)
    if n == 1:
        return x

    # Recursion
    even = fft_rec(x[::2])
    odd = fft_rec(x[1::2])

    # Combine results
    result = [0] * n
    for k in range(n // 2):
        t = cmath.exp(-2j * cmath.pi * k / n) * odd[k]
        result[k] = even[k] + t
        result[k + n // 2] = even[k] - t

    return result


def ifft_rec(x):
    # Base case , if the length of our input is one
    len_vec = len(x)
    if len_vec == 1:
        return x

    # Recursion
    odd = ifft_rec(x[1::2])
    even = ifft_rec(x[::2])

    # Combine results
    result = [0] * len_vec
    for k in range(len_vec // 2):
        t = cmath.exp(2j * cmath.pi * k / len_vec) * odd[k]
        result[k] = (even[k] + t) / 2
        result[k + len_vec // 2] = (even[k] - t) / 2

    return result


"""# task 2"""


def fft_1d_radix2(vec, direction):
    len_vec = len(vec)

    if is_power_of_two(len_vec) == False:
        return ValueError('the input isnt power of 2 ')

    if direction == 'FFTW_FORWARD':
        return fft_bit(vec)

    if direction == 'FFTW_BACKWARD':
        return ifft_bit(vec)


def fft_bit(x):
    n = len(x)
    # calculate the log (base 2) of the length of our input
    m = int(np.log2(n))

    # initialize an array of complex number , the length is lik our input
    y = np.zeros(n, dtype=complex)

    # a bit reverse the input
    for i in range(n):
        rev = 0
        for j in range(m):
            rev |= ((i >> j) & 1) << (m - 1 - j)
        y[rev] = x[i]

    # compute the FFT
    for s in range(1, m + 1):
        w_n = np.exp(-2j * np.pi / (2 ** s))
        for k in range(0, n, 2 ** s):
            w = 1
            for j in range(0, 2 ** (s - 1)):
                t = w * y[k + j + 2 ** (s - 1)]
                u = y[k + j]
                y[k + j] = u + t
                y[k + j + 2 ** (s - 1)] = u - t
                w *= w_n

    return y


def ifft_bit(x):
    n = len(x)
    # calculate the log (base 2) of the length of our input
    m = int(np.log2(n))

    # initialize an array of complex number , the length is lik our input
    y = np.zeros(n, dtype=complex)

    # a bit reverse the input
    for i in range(n):
        rev = 0
        for j in range(m):
            rev |= ((i >> j) & 1) << (m - 1 - j)
        y[rev] = x[i]

    # compute the FFT
    for s in range(1, m + 1):
        w_n = np.exp(2j * np.pi / (2 ** s))
        for k in range(0, n, 2 ** s):
            w = 1
            for j in range(0, 2 ** (s - 1)):
                t = w * y[k + j + 2 ** (s - 1)]
                u = y[k + j]
                y[k + j] = u + t
                y[k + j + 2 ** (s - 1)] = u - t
                w *= w_n

    return y / n


# HW 3

def fft_1d(vec, direction):
    if direction == 'FFTW_FORWARD':
        return bluestein_fft(vec)

    if direction == 'FFTW_BACKWARD':
        return bluestein_ifft(vec)


def bluestein_fft(vec):
    len_vec = len(vec)

    # define padding length
    padding_size = 2 ** int(np.ceil(np.log2(2 * len_vec - 1)))

    k = np.arange(len_vec)
    y = np.exp(np.pi * 1j * k ** 2 / len_vec)
    x = np.conj(y)

    # padding x ,y by formula
    padding_x = np.concatenate((vec * y, np.zeros(padding_size - len_vec)))
    padding_y = np.concatenate((x, np.zeros(padding_size - 2 * len_vec + 1), x[1:]))

    # convolution
    dft_x = dft_1d_matrix(padding_x, 'FFTW_FORWARD')
    dft_y = dft_1d_matrix(padding_y, 'FFTW_FORWARD')

    convolution = dft_1d_matrix(dft_x * dft_y, 'FFTW_BACKWARD')[:len_vec]

    result = convolution * y

    return result


def bluestein_ifft(vec):
    len_vec = len(vec)

    padding_size = 2 ** int(np.ceil(np.log2(2 * len_vec - 1)))

    multi_x = np.zeros(len_vec, dtype=complex)
    multi_y = np.zeros(len_vec, dtype=complex)

    for j in range(len_vec):
        multi_x[j] = np.exp((1j * np.pi * (j ** 2)) / len_vec)
        multi_y[j] = np.exp((-1j * np.pi * (j ** 2)) / len_vec)

    # padding x,y
    padding_x = np.concatenate((vec * multi_y, np.zeros(padding_size - len_vec)))
    padding_y = np.concatenate((multi_x, np.zeros(padding_size - (2 * len_vec) + 1), multi_x[1:]))

    dft_x = np.array(dft_forward(padding_x))
    dft_y = np.array(dft_forward(padding_y))

    # calculate convolution
    convolution = dft_backward(dft_x * dft_y)[:len_vec]

    result = (convolution * multi_y) / len_vec

    return result


# HW 4

def fft_1d_real(vec, direction):
    if np.all(np.isreal(vec)):
        if direction == "FFTW_FORWARD":
            return fft_real(vec)

        if direction == "FFTW_BACKWARD":
            return ifft_real_1(vec)


def fft_real(vec):
    len_vec = int(len(vec))
    h = np.zeros(len_vec // 2, dtype=complex)
    for j in range(len_vec // 2):
        h[j] += vec[2 * j] + 1j * vec[2 * j + 1]

    # I have mistaken at the implementation at fft_1d ,so I use the build in function
    # compute dft on h, "H"
    #  h_fft = np.fft.fft(h)
    h_fft = fft_1d(h, 'FFTW_FORWARD')

    h_fft_conj = np.conjugate(h_fft)

    result = np.zeros(len_vec // 2 + 1, dtype=complex)

    # we know result[0] and result[len//2] by formula
    result[0] = np.sum(vec)

    for k in range(len_vec // 2):
        result[len_vec // 2] += vec[2 * k] - vec[2 * k + 1]

    # calculate result[k] by formula, we now calculate the result for first n//2 values
    for k in range(1, len_vec // 2):
        result[k] = 0.5 * (h_fft[k] + h_fft_conj[len_vec // 2 - k]) - (0.5j * (
                h_fft[k] - h_fft_conj[len_vec // 2 - k]) * (np.exp(2 * np.pi * 1j * k / len_vec)))

    # add result and inverse result (conjugate)
    result = np.concatenate((result, np.conjugate(result[1:len_vec // 2][::-1])))

    return result


def ifft_real_1(vec):
    len_vec = int(len(vec))
    ake = np.zeros(len_vec // 2, dtype=complex)
    ako = np.zeros(len_vec // 2, dtype=complex)

    vec_conj = np.conjugate(vec)

    for k in range(len_vec // 2):
        ake[k] = 0.5 * (vec[k] + vec_conj[len_vec // 2 - k])
        ako[k] = 0.5 * np.exp(-2j * np.pi * k / len_vec) * (vec[k] - vec_conj[len_vec // 2 - k])

    h = ake + np.multiply(1j, ako)

    result = np.zeros(len_vec, dtype=complex)

    #    h_ifft = fft_1d(h,'FFTW_BACKWARD')

    h_ifft = np.fft.ifft(h)

    for i in range(len_vec // 2):
        result[2 * i] = np.real(h_ifft[i])
        result[2 * i + 1] = np.imag(h_ifft[i])

    return result


# task 2 - compare fft_1d_real and fft_1d

def compare_running_time():
    # range of input sizes to test
    input_sizes = np.arange(2, 30, 4)
    fft_1d_times = []
    fft_1d_real_times = []

    # Measure the running times for each input size
    for size in input_sizes:
        # generate a random input signal
        vec = np.random.rand(size)
        start_time = time.time()
        fft_1d(vec, "FFTW_FORWARD")
        fft_1d_times.append(time.time() - start_time)

        start_time = time.time()
        fft_1d_real(vec, "FFTW_BACKWARD")
        fft_1d_real_times.append(time.time() - start_time)

    # Plot the running time data
    plt.plot(input_sizes, fft_1d_times, label="fft_1d")

    plt.plot(input_sizes, fft_1d_real_times, label="fft_1d_real")

    plt.xlabel("n")
    plt.ylabel("Running time (s)")
    plt.legend()
    plt.show()


# compare_running_time()

# hw 5

def fft_1d_fftw(vec, direction):
    if direction == 'FFTW_FORWARD':
        return fft.fft(vec)

    if direction == 'FFTW_BACKWARD':
        return fft.ifft(vec)


def fft_1d_real_fftw(vec, direction):
    if direction == 'FFTW_FORWARD':
        return fft.rfft(vec)

    if direction == 'FFTW_BACKWARD':
        return fft.irfft(vec)


def compare_to_fftw():
    # calculate the relative error for fft_1d and fft_1d_fftw

    n_values = np.arange(2, 300, 4)

    directions = ['FFTW_FORWARD', 'FFTW_BACKWARD']

    relative_error_by_n_forward = None
    relative_error_by_n_backward = None

    for direction in directions:
        relative_error_by_n = []
        ffd_1d_times = []
        fft_1d_fftw_times = []
        for n in n_values:
            vec = np.random.rand(n) + 1j * np.random.rand(n)
            start_time = time.time()
            fft_1d_vec = fft_1d(vec, direction)
            end_time = time.time()
            ffd_1d_times.append(end_time - start_time)

            start_time = time.time()
            fft_1d_fftw_vec = fft_1d_fftw(vec, direction)
            end_time = time.time()
            fft_1d_fftw_times.append(end_time - start_time)

            arr = np.append(fft_1d_vec[0], fft_1d_vec[1:][::-1])
            fft_1d_vec = np.array(arr)

            relative_error = np.linalg.norm(fft_1d_vec - fft_1d_fftw_vec) / np.linalg.norm(fft_1d_vec)
            relative_error_by_n.append(relative_error)

        if direction == 'FFTW_FORWARD':
            relative_error_by_n_forward = relative_error_by_n
        else:
            relative_error_by_n_backward = relative_error_by_n

    plt.plot(n_values, relative_error_by_n_forward, label='forward')
    plt.plot(n_values, relative_error_by_n_backward, label='backward')

    plt.plot(n_values, ffd_1d_times, label='time of function by n of fft_1d')
    plt.plot(n_values, fft_1d_fftw_times, label='time of function by n of fft_1d_fftw')

    plt.xlabel('n')
    plt.ylabel('Relative error')
    plt.title('Relative error versus n of fft_1d and fft_1d_fftw :')
    plt.legend()
    plt.show()

    # comparing of fft for real numbers

    for direction in directions:

        relative_error_by_n = []
        ffd_1d_real_times = []
        fft_1d_real_fftw_times = []

        for n in n_values:
            vec = np.random.rand(n)
            start_time = time.time()
            fft_1d_real_vec = fft_1d_real(vec, direction)
            end_time = time.time()
            ffd_1d_real_times.append(end_time - start_time)

            start_time = time.time()
            fft_1d_real_fftw_vec = fft_1d_real_fftw(vec, direction)
            end_time = time.time()
            fft_1d_real_fftw_times.append(end_time - start_time)

            if direction == 'FFTW_FORWARD':
                fft_1d_real_vec = np.conjugate(fft_1d_real_vec)

            relative_error = np.linalg.norm(
                fft_1d_real_vec[:len(fft_1d_real_fftw_vec)] - fft_1d_real_fftw_vec[
                                                              :len(fft_1d_real_vec)]) / np.linalg.norm(fft_1d_real_vec)
            relative_error_by_n.append(relative_error)

        if direction == 'FFTW_FORWARD':
            relative_error_by_n_forward = relative_error_by_n
        else:
            relative_error_by_n_backward = relative_error_by_n

    plt.plot(n_values, relative_error_by_n_forward, label='forward')
    plt.plot(n_values, relative_error_by_n_backward, label='backward')

    plt.plot(n_values, ffd_1d_real_times, label='time by n of fft_1d_real')
    plt.plot(n_values, fft_1d_real_fftw_times, label='time by n of fft_1d_real_fftw')

    plt.xlabel('n')
    plt.ylabel('Relative error')
    plt.title('Relative error versus n')
    plt.legend()
    plt.show()


# testing

# compare_to_fftw()


FFTW_FORWARD = 'FFTW_FORWARD'
FFTW_BACKWARD = 'FFTW_BACKWARD'


# HW 6
# two-dimensional FFT

# task 1
def fft_2d(arr, direction):
    if direction == 'FFTW_FORWARD':
        return fft_2d_forward(arr)

    if direction == 'FFTW_BACKWARD':
        return fft_2d_backward(arr)


def fft_2d_forward(arr):
    len_arr_rows = arr.shape[0]
    len_arr_cols = arr.shape[1]

    result = np.zeros((len_arr_rows, len_arr_cols), dtype=complex)

    kk, jj = np.meshgrid(np.arange(len_arr_cols), np.arange(len_arr_rows))

    # Calculate the Fourier transform
    # rows
    for m in range(len_arr_rows):
        # columns
        for n in range(len_arr_cols):
            result[m, n] = np.sum(arr * np.exp(-2j * np.pi * (jj * m / len_arr_rows + kk * n / len_arr_cols)))

    return result


def fft_2d_backward(arr):
    len_arr_rows = arr.shape[0]
    len_arr_cols = arr.shape[1]

    result = np.zeros((len_arr_rows, len_arr_cols), dtype=complex)

    kk, jj = np.meshgrid(np.arange(len_arr_cols), np.arange(len_arr_rows))

    # Calculate the Fourier transform
    # rows
    for m in range(len_arr_rows):
        # columns
        for n in range(len_arr_cols):
            result[m, n] = (np.sum(arr * np.exp(-2j * np.pi * (jj * m / len_arr_rows + kk * n / len_arr_cols)))) / (
                    len_arr_rows * len_arr_cols)
        # result[m, n] = result[m, n] / (len_arr_rows * len_arr_cols)

    return np.conjugate(result)


# test for task 1
arr = np.random.rand(5, 7) * 10
test1 = ifft2(arr)
test2 = fft_2d_backward(arr)


# task 2

def fft_1d_fftw(vec, direction):
    vec_len = len(vec)
    if direction == 'FFTW_FORWARD':
        return fft.fft(vec) * vec_len
    if direction == 'FFTW_BACKWARD':
        return fft.ifft(vec) / vec_len


def fft_2d_for_image(arr, direction):
    # Check if the input array is square or rectangular
    def check_rectangular(arr):
        a = arr.shape[0]
        b = arr.shape[1]
        if a == b:
            return False
        else:
            return True

    # Pad the input array to the nearest power of 2 if rectangular
    if check_rectangular:
        padding_size = max(arr.shape)
        padded_arr = np.zeros((padding_size, padding_size), dtype=complex)
        padded_arr[:arr.shape[0], :arr.shape[1]] = arr
    else:
        padded_arr = arr

    # Perform 1D FFT along the rows
    fft_rows = np.zeros(padded_arr.shape, dtype=complex)
    for i in range(padded_arr.shape[0]):
        fft_rows[i, :] = fft_1d_fftw(padded_arr[i, :], direction)

    # Perform 1D FFT along the columns
    fft_col = np.zeros(padded_arr.shape, dtype=complex)
    for j in range(padded_arr.shape[1]):
        fft_col[:, j] = fft_1d_fftw(fft_rows[:, j], direction)

    # Return the resulting transformed array
    if check_rectangular:
        return fft_col[:arr.shape[0], :arr.shape[1]]
    else:
        return fft_col


def cfft_2d_for_image(arr, direction):
    m, n = arr.shape

    shifted_arr = np.roll(np.roll(arr, -n // 2, axis=1), -m // 2, axis=0)
    result = fft_2d_for_image(shifted_arr, direction)

    return np.roll(np.roll(result, n // 2, axis=1), m // 2, axis=0)


def cfft_2d(arr, direction):
    if direction == 'FFTW_FORWARD':
        return cfft_2d_forward(arr)

    if direction == 'FFTW_BACKWARD':
        return cfft_2d_backward(arr)


def cfft_2d_forward(arr):
    len_arr_rows, len_arr_cols = arr.shape

    result = np.empty((len_arr_rows, len_arr_cols), dtype=complex)

    # Calculate the Fourier transform
    # rows
    for m in range(len_arr_rows):
        # columns
        for n in range(len_arr_cols):
            # Create 2D arrays of indices for the frequency domain
            uu, vv = np.indices(result.shape)
            mu = uu - len_arr_rows // 2
            nu = vv - len_arr_cols // 2

            # Calculate the Fourier transform using the formula
            result[m, n] = np.sum(arr * np.exp(2j * np.pi * (mu * m / len_arr_rows + nu * n / len_arr_cols)))

        # check with ido about the minus and the conj which I did for the values in the odds places
        '''   
    for i in range(len_arr_rows):
        for j in range(1, len_arr_cols, 2):
            result[i, j] = np.conj(-result[i, j])
            '''

    result = np.conjugate((-result))

    return result


def cfft_2d_backward(arr):
    len_arr_rows, len_arr_cols = arr.shape

    result = np.empty((len_arr_rows, len_arr_cols), dtype=complex)

    # Calculate the Fourier transform
    # rows
    for m in range(len_arr_rows):
        # columns
        for n in range(len_arr_cols):
            # Create 2D arrays of indices for the frequency domain
            uu, vv = np.indices(result.shape)
            mu = uu - len_arr_rows // 2
            nu = vv - len_arr_cols // 2

            # Calculate the Fourier transform using the formula
            result[m, n] = (np.sum(arr * np.exp(-2j * np.pi * (mu * m / len_arr_rows + nu * n / len_arr_cols)))) / (
                    len_arr_rows * len_arr_cols)

    # check with ido about the minus and the conj which I did for the values in the odds places

    for i in range(len_arr_rows):
        for j in range(1, len_arr_cols, 2):
            result[i, j] = np.conj(-result[i, j])

    return result


# test
# make array and compare our implementation to the build in function of python

row1 = np.array([-10, -1, 1, 10])

# Stack the two arrays vertically to create a two-dimensional array
arr = np.vstack([row1, row1, row1, row1, row1, row1, row1, row1])

test1 = ifft2(arr)

test2 = cfft_2d_backward(arr)

result_test_1 = np.allclose(test1, test2, rtol=1e-1, atol=1e-1)



# HW 7

def verify_input_2d(input_list):
    """
    Verifies that the input list contains only complex or real numbers.

    Args:
    input_list (numpy array or list): The list to be verified.(2D)

    """
    # check that the vector is iterable
    if not isinstance(input_list,np.ndarray) :
        input_list = np.array(input_list)
    try:
        iter(input_list)
    except TypeError as ex:
        raise ValueError('Input vector is not iterable') from ex
    for row in input_list:
        for item in row:
            # check for wrong dtypes
            if not isinstance(item, (complex, float, int,np.uint8,np.int32)):
                raise ValueError('Input vector contains variables that arent sint,float or complex')
    return input_list



def shift_array(arr,shift):
    'helper function to shift 1D array'
    if not isinstance(arr,np.ndarray):
        arr = np.array(arr)

    vec_len = len(arr)
    arr = arr.reshape((1,vec_len))

    # Perform 2D complex Fast Fourier Transform in forward direction
    fft_mat = cfft_2d(arr,'FFTW_FORWARD')

    # Calculate the shifting factor using exponential function
    factor = np.exp(-2*np.pi*1j*np.arange(vec_len)*shift/vec_len).reshape(1,vec_len)

    # Apply the shifting factor to the Fourier transformed array
    fft_mat = fft_mat*factor

    # Perform 2D complex Fast Fourier Transform in backward direction
    fft_mat = cfft_2d(fft_mat,'FFTW_BACKWARD')

    return fft_mat

def pad_img(img):
    'padding image'
    img_pad = np.pad(img, img.shape)
    return img_pad

def shift_tester():
    'Test if the shift array works correctly using analytical calculations'

    arr_1 = [1,2,3,4,5,6,7,8,9]
    arr_1_shift  = [6,7,8,9,1,2,3,4,5]
    arr_2 = [1j,1,1+1j,2+1j,4,5]
    arr_2_shift = [4,5,1j,1,1+1j,2+1j]

    print(np.abs(shift_array(arr_1,4)))
    print((shift_array(arr_2, 2)))

    bool_1 = np.allclose(shift_array(arr_1,4), arr_1_shift)
    bool_2 = np.allclose(shift_array(arr_2,2), arr_2_shift)

    if (bool_1 is False or bool_2 is False):
        print('shift_array function ERROR')
    else:
        print("shift_array function works")

def rotate_90_deg_mul(image, angle):
    'manuel rotation of the image before rotate in angle less than 90'
    if angle not in {0, 90, 180, 270}:
        raise ValueError('angle is not one of {0, 90, 180, 270}')
    if angle == 0:
        return image
    if angle == 90:
        return image.T[::-1]
    if angle == 180:
        return image[::-1][:, ::-1]
    if angle == 270:
        return image.T[:, ::-1]
    return 'angle is not one of {0, 90, 180, 270}'
def rotate(image,angle,padded=False):
    'rotate image using 2D fft'
    image = verify_input_2d(image)
    # Pad the image based on the desired rotation angle
    if padded is False:
        image = pad_img(image)

    if angle > np.pi/2:
        angle_mod = angle - (angle%(np.pi/2))
        angle_mod = angle_mod/np.pi *180
        angle = angle%(np.pi/2)
        image =  rotate_90_deg_mul(image,angle_mod)

    angle= - angle

    rows,_ = image.shape

    # Perform row-wise shifting for the top half of the image
    for row in range(rows):
        image[row] = np.abs(shift_array(image[row],(row-rows/2)*np.tan(angle/2)))

    # Transpose the image and perform row-wise shifting for the bottom half of the image
    image = image.T
    for row in range(rows):
        image[row] = np.abs(shift_array(image[row],(row-rows/2)*np.tan(-angle/2)))
    image = image.T

    # Perform row-wise shifting for the entire image
    for row in range(rows):
        image[row] = np.abs(shift_array(image[row,:],(row-rows/2)*np.tan(angle/2)))

    return image


def pil_rotate_image(image_array, angle,padded=False):
    'rotate image using PIL'
    if padded is False:
        image_array = pad_img(image_array)

    # Convert NumPy array to PIL Image
    image = Image.fromarray(image_array)
    angle = angle/np.pi * 180
    # Rotate the image by the specified angle
    rotated_image = image.rotate(angle)

    # Convert the rotated image back to NumPy array
    rotated_array = np.array(rotated_image)

    return rotated_array

def task_2_compare(img,angle):
    "comparing PIL and FFT rotation"
    # Plot the image
    plt.imshow(img, cmap='gray')
    plt.show()

    # rotate the image with fft and PIL
    fft_rotation  = rotate(img,angle)
    pil_rotation = pil_rotate_image(img,angle)
    plt.imshow(np.hstack((fft_rotation, pil_rotation)), cmap='gray')
    plt.title('FFT rotation VS  Pil rotation')
    plt.show()

    fft_rotation_rec = rotate(fft_rotation.copy(),-angle,True)
    pil_rotation_rec = pil_rotate_image(pil_rotation.copy(),-angle,True)
    plt.imshow(np.hstack((fft_rotation_rec, pil_rotation_rec)), cmap='gray')
    plt.title('FFT Rec VS  Pil Rec')
    plt.show()
    padded_image = pad_img(img)
    diff_fft = fft_rotation_rec - padded_image
    diff_pil = pil_rotation_rec - padded_image

    fft_mse = ((diff_fft)**2).mean()
    pil_mse = ((diff_pil)**2).mean()

    fft_ssim = ssim(fft_rotation_rec,padded_image)
    pil_ssim = ssim(pil_rotation_rec,padded_image)

    plt.imshow(diff_fft,cmap='gray')
    plt.title('Difference between original and reconstruct with FFT rotation')
    plt.show()
    plt.imshow(diff_pil,cmap='gray')
    plt.title('Difference between original and reconstruct with PIL rotation')
    plt.show()

    print(f'PIl MSE: {pil_mse}')
    print(f'PIl SSIM: {pil_ssim}')
    print(f'FFT MSE: {fft_mse}')
    print(f'FFT SSIM: {fft_ssim}')


# hw 8
import os
from PIL import Image
import numpy as np

from scipy.fftpack import dct
import zlib
from scipy.fftpack import idct

import math
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
import os

import argparse

def split_image_into_blocks(image):
    width, height = image.size
    blocks = []

    # Calculate the number of blocks in the horizontal and vertical directions
    num_blocks_x = width // 8
    num_blocks_y = height // 8

    # Iterate over each block and extract the pixel values
    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            # Define the coordinates for the current block
            x = j * 8
            y = i * 8
            box = (x, y, x + 8, y + 8)

            # Extract the pixel values within the block region
            block = image.crop(box)
            block_pixels = np.array(block)

            # Append the block to the list
            blocks.append(block_pixels)

    return blocks




def subtract_128(blocks):
    adjusted_blocks = []

    for block in blocks:
        adjusted_block = block.astype(np.int16) - 128
        adjusted_blocks.append(adjusted_block)

    return adjusted_blocks


# my implementation for dct
def build_custom_matrix(size):
    custom_matrix = np.zeros((size, size))
    for i in range(size):
        if i==0 :
            custom_matrix[i, :] = np.sqrt(2)/2
        else:

            custom_matrix[i, :] = np.cos(np.pi * np.linspace((i)/16, (i)/16 + ((i)*2/16)*(size-1), size))
    return custom_matrix * (0.5)

# Example usage


dct_forward = build_custom_matrix(8)


dct_backward = np.linalg.inv(dct_forward)



# build in dct
def dct_2d(block):

    '''
    block = block.astype(np.float32)  # Convert block to float for precision
    transformed_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
    return transformed_block
    '''
    return np.dot(dct_forward,block)


'''
# Example usage
block = np.random.randint(0, 256, size=(8, 8))  # Random 8x8 block for demonstration

block = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92], [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])

dct_result = dct_2d(block)
dct_build_in = dct_2d_builtin(block)

'''

def create_quantization_table(quality):
    # Quality ranges from 1 to 100, higher values indicate lower compression (better quality)
    if quality < 1:
        quality = 1
    if quality > 100:
        quality = 100

    # Define the quantization matrix
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality

    q_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    if quality >= 100:
        quantization_table = np.full((8, 8), 1)

    else:
        quantization_table = np.round((scale * q_matrix + 50) / 100)

    return quantization_table





def divide_block_by_quantization_table(block, quantization_table):
    return np.divide(block, quantization_table).astype(np.int32)


def zigzag(block):
    rows, columns = block.shape
    solution = [[] for _ in range(rows + columns - 1)]

    for i in range(rows):
        for j in range(columns):
            _sum = i + j
            if _sum % 2 == 0:
                solution[_sum].insert(0, block[i][j])
            else:
                solution[_sum].append(block[i][j])

    # Flatten the solution into a 1D array
    zigzag_order = [element for sublist in solution for element in sublist]
    return np.array(zigzag_order)



def compress_vectors(vectors):
    compressed_data = zlib.compress(bytes(vectors))
    return compressed_data




def encode(input_image_filename, output_compressed_filename, quality):
    image_path = input_image_filename
    image = Image.open(image_path)

    # print(image)
    image_shape = image.size
    # print("Image shape:", image_shape)

    image_blocks = split_image_into_blocks(image)

    adjusted_blocks = subtract_128(image_blocks)

    transform_dct2 = []

    quantization_table = create_quantization_table(quality)

    for block in adjusted_blocks:
        dct2_block = dct_2d(block)

        divided_block = divide_block_by_quantization_table(dct2_block, quantization_table)

        zigzag_block = zigzag(divided_block)

        compressed_block = compress_vectors(zigzag_block)

        transform_dct2.append(compressed_block)


    with open(output_compressed_filename, 'w') as f:
        for s in transform_dct2:
            f.write(str(s) + '\n')

    return transform_dct2



def decompress_data(compressed_data):
    # Decompress the data


    len_vec = len(compressed_data)

    compressed_data = compressed_data[2:len_vec-1]

    compressed_data_bytes = bytes(compressed_data, 'latin-1').decode('unicode-escape').encode('latin-1')

    decompressed_data = zlib.decompress(compressed_data_bytes)
   # print('decompressed_data',decompressed_data)
    # Convert the decompressed data to a vector of numbers
    vector = np.frombuffer(decompressed_data, dtype=np.int32)

    return vector


def reverse_zigzag(vector):
    # Create an empty matrix
    matrix = np.zeros((8, 8))

    # Initialize variables for tracking row and column indices
    row, col = 0, 0

    # Zigzag pattern indices
    zigzag_indices = [(0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
                      (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
                      (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
                      (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
                      (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
                      (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
                      (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
                      (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)]

    # Assign the vector values to the matrix in the reverse zigzag pattern
    for i, value in enumerate(vector):
        row_idx, col_idx = zigzag_indices[i]
        matrix[row_idx, col_idx] = value

    return matrix


def reverse_quantization_matrix(matrix):
    quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                    [12, 12, 14, 19, 26, 58, 60, 55],
                                    [14, 13, 16, 24, 40, 57, 69, 56],
                                    [14, 17, 22, 29, 51, 87, 80, 62],
                                    [18, 22, 37, 56, 68, 109, 103, 77],
                                    [24, 35, 55, 64, 81, 104, 113, 92],
                                    [49, 64, 78, 87, 103, 121, 120, 101],
                                    [72, 92, 95, 98, 112, 100, 103, 99]])

    result = np.multiply(matrix, quantization_matrix)
    return result


def inverse_dct(matrix):

    '''
    # Apply the inverse discrete cosine transform (IDCT) using scipy's idct function
    idct_matrix = idct(idct(matrix.T, norm='ortho').T, norm='ortho')
    '''

    return np.dot(dct_backward,matrix)


def add_128(matrix):
    # Add 128 to each element of the matrix
    result = matrix + 128

    return result




def combine_blocks(blocks, image_shape):
    num_rows, num_cols = image_shape
    block_height = 8
    block_width = 8

    # Calculate the number of blocks in each dimension
    num_blocks_rows = num_rows // block_height
    num_blocks_cols = num_cols // block_width

    # Initialize the full image matrix
    full_image = np.zeros(image_shape)

    # Iterate over each block
    for block_row in range(num_blocks_rows):
        for block_col in range(num_blocks_cols):
            # Calculate the starting row and column indices of the current block
            start_row = block_row * block_height
            start_col = block_col * block_width

            # Retrieve the current block
            block = blocks[block_row * num_blocks_cols + block_col]

            # Assign the block values to the corresponding region in the full image
            full_image[start_row:start_row + block_height, start_col:start_col + block_width] = block

    return full_image


def show_image(matrix):
    # Show the reconstructed image
    plt.imshow(matrix, cmap='gray')
    plt.axis('off')
    plt.show()


def split_compressed_data(compressed_data):
    # Split the compressed data based on 'b'
    compressed_rows = compressed_data.split('b')

    # Remove any empty rows
    compressed_rows = [row for row in compressed_rows if row]

    return compressed_rows


def save_decompressed_data(data, output_file):
    with open(output_file, 'w') as file:
        # Convert the ndarray to a string representation
        data_str = np.array2string(data)

        # Write the string representation to the file
        file.write(data_str)





def decode(input_compressed_filename, output_image_filename):
    file_path = input_compressed_filename  # Replace with the path to your text file

    with open(input_compressed_filename, 'r') as f:
        matrix = [line.rstrip('\n') for line in f]


    # Initialize an empty list to store the reconstructed blocks
    reconstructed_blocks = []
    counter = 1
    # Iterate over each compressed row


    for compressed_row in matrix:

        temp = decompress_data(compressed_row)

        temp = reverse_zigzag(temp)
        temp = reverse_quantization_matrix(temp)
        temp = inverse_dct(temp)
        temp = add_128(temp)

        reconstructed_blocks.append(temp)



    # print(reconstructed_blocks[0])

    reconstructed_image = combine_blocks(reconstructed_blocks, (512, 512))
    reconstructed_image = np.array(reconstructed_image)


    # there is a problem with the saving , not save as a matrix
    save_decompressed_data(reconstructed_image.astype(np.int32), output_image_filename)

    return reconstructed_image.astype(np.int32)





saving = "C:\\Users\\matan\\Downloads\\matan.txt"


# Calculate PSNR
def psnr_build_in(original_image, compressed_image):
    psnr_value = peak_signal_noise_ratio(original_image, compressed_image)

    return psnr_value


def mean_squared_error_my_function(image1, image2):
    # Ensure the images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same shape")

    # Calculate the squared difference between the images
    diff = image1 - image2
    squared_diff = np.square(diff)

    # Calculate the mean squared error
    mse = np.mean(squared_diff)

    return mse


def calculate_psnr(original_image_matrix, reconstructed_image_matrix):
    # Convert the images to numpy arrays
    original_image = np.array(original_image_matrix, dtype=np.float64)
    reconstructed_image = np.array(reconstructed_image_matrix, dtype=np.float64)

    # Compute the mean squared error (MSE)

    mse = mean_squared_error_my_function(original_image, reconstructed_image)

    # Calculate the maximum pixel value based on the bit depth
    max_value = np.max(original_image)

    # Calculate the PSNR
    psnr = 20 * math.log10(max_value) - 10 * math.log10(mse)

    return psnr

'''
file1 = "C:\\Users\\matan\\Downloads\\3.gif"
file2 = "C:\\Users\\matan\\Downloads\\1.gif"
file3 = "C:\\Users\\matan\\Downloads\\47.gif"
file4 = "C:\\Users\\matan\\Downloads\\12.gif"

image_filenames = [file1, file2, file3, file4]
'''



quality_levels = [50, 25, 10, 5]

'''
saving2 = "C:\\Users\\matan\\Downloads\\saving2.txt"
'''


def get_file_size(file_path):
    if os.path.isfile(file_path):
        return os.path.getsize(file_path)
    else:
        return -1




def calculate_compression_ratio(original_size, compressed_size):
    ratio = (compressed_size / original_size) * 100
    return ratio



def task_2_images(image_filenames,saving2):
    for i, image_filename in enumerate(image_filenames):
        # Load the original image
        original_image = plt.imread(image_filename)

        # Create subplots for original and reconstructed images

        fig, axs = plt.subplots(2, len(quality_levels) + 1, figsize=(12, 6))
        axs[0, 0].imshow(original_image, cmap='gray')
        axs[0, 0].set_title('Original')

        # Get the size of the image file in bytes
        original_size = get_file_size(image_filename)

        # Iterate over quality levels and perform compression
        for j, quality in enumerate(quality_levels):
            # Compress the image
            compressed_image = encode(image_filename, saving, quality)

            size_compressed = get_file_size(saving)

            reatio_bytes = size_compressed/original_size

            # pay attention which the input_image_filename isn't care us now, because we get
            # as input thr compressed image
            reconstructed_image = decode(saving,saving2)

            # Calculate PSNR
            psnr = peak_signal_noise_ratio(original_image, reconstructed_image)

            # Calculate compressed image size
            # compressed_size = os.path.getsize(file3)

            # Calculate compression ratio
            #compression_ratio = calculate_compression_ratio(original_size, compressed_size)

            # Plot the reconstructed image
            axs[1, j + 1].imshow(reconstructed_image, cmap='gray')
            axs[1, j + 1].set_title(f'Quality: {quality}\nPSNR: {psnr:.2f}\nCompression Ratio: {reatio_bytes:.1f}%')

        # Remove axis labels
        for ax in axs.flat:
            ax.axis('off')

        # Save the plot
        plt.tight_layout()
        plt.savefig(f'image_{i + 1}_plot.png')
        plt.show()
        plt.close()

#task_2_images(image_filenames,saving2)



if __name__ == "__main__":
    # Create the command-line argument parser
    parser = argparse.ArgumentParser(description="Image compression command-line tool")

    # Add subparsers for encode and decode commands
    subparsers = parser.add_subparsers(dest="command")

    # Encode command
    encode_parser = subparsers.add_parser("encode", help="Encode an image")
    encode_parser.add_argument("input_image", help="Input image filename")
    encode_parser.add_argument("compressed_file", help="Compressed file output")
    encode_parser.add_argument("quality", type=int, help="Compression quality")

    # Decode command
    decode_parser = subparsers.add_parser("decode", help="Decode an image")
    decode_parser.add_argument("compressed_file", help="Compressed file input")
    decode_parser.add_argument("output_image", help="Output image filename")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Perform the specified command
    if args.command == "encode":
        encode(args.input_image, args.compressed_file, args.quality)
    elif args.command == "decode":
        decode(args.compressed_file, args.output_image)
