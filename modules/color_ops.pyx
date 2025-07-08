# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

import numpy as np
cimport numpy as np
from cython.parallel import prange
from modules.benchmark import benchmark

# Define types for faster operations
ctypedef np.float32_t DTYPE_t
ctypedef np.uint8_t UINT8_t

@benchmark("apply_color_parallel")
def apply_color_parallel(np.ndarray[DTYPE_t, ndim=3] pixel_buffer,
                         np.ndarray[np.int32_t, ndim=1] masked_indices,
                         np.ndarray[DTYPE_t, ndim=1] glow_masked,
                         np.ndarray[DTYPE_t, ndim=1] color_float):
    """
    Apply color to pixels in parallel.

    Parameters:
    -----------
    pixel_buffer : ndarray
        3D array of shape (height, width, 3) for RGB pixel data
    masked_indices : ndarray
        1D array of indices where mask is True
    glow_masked : ndarray
        1D array of glow factors for masked pixels
    color_float : ndarray
        RGB color values as floats in range [0, 1]
    """
    cdef int num_pixels = masked_indices.shape[0]
    cdef int i, row, col
    cdef int height = pixel_buffer.shape[0]
    cdef int width = pixel_buffer.shape[1]
    cdef DTYPE_t r = color_float[0]
    cdef DTYPE_t g = color_float[1]
    cdef DTYPE_t b = color_float[2]
    cdef DTYPE_t glow

    # Simple loop without OpenMP for now
    for i in range(num_pixels):
        idx = masked_indices[i]
        # Convert flat index to 2D coordinates
        row = idx // width
        col = idx % width
        glow = glow_masked[i]

        # Ensure we're within bounds
        if 0 <= row < height and 0 <= col < width:
            pixel_buffer[row, col, 0] = glow * r
            pixel_buffer[row, col, 1] = glow * g
            pixel_buffer[row, col, 2] = glow * b

    return pixel_buffer

@benchmark("apply_spectrogram_color")
def apply_spectrogram_color(np.ndarray[DTYPE_t, ndim=3] spectrogram_data,
                           np.ndarray[DTYPE_t, ndim=1] color_float,
                           float intensity_scale=1.0):
    """
    Apply color to spectrogram data in parallel using OpenMP.

    Parameters:
    -----------
    spectrogram_data : ndarray
        3D array of shape (height, width, 3) for RGB spectrogram data
    color_float : ndarray
        RGB color values as floats in range [0, 1]
    intensity_scale : float
        Scale factor for intensity values
    """
    cdef int height = spectrogram_data.shape[0]
    cdef int width = spectrogram_data.shape[1]
    cdef int i, j
    cdef DTYPE_t r = color_float[0]
    cdef DTYPE_t g = color_float[1]
    cdef DTYPE_t b = color_float[2]
    cdef DTYPE_t intensity

    # Use OpenMP to parallelize the color application
    with nogil:
        for i in prange(height, schedule='static'):
            for j in range(width):
                intensity = spectrogram_data[i, j, 0] * intensity_scale
                spectrogram_data[i, j, 0] = intensity * r
                spectrogram_data[i, j, 1] = intensity * g
                spectrogram_data[i, j, 2] = intensity * b

    return spectrogram_data
