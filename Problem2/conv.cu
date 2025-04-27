/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This sample demonstrates how use texture fetches in CUDA
 *
 * This sample takes an input PGM image (image_filename) and generates
 * an output PGM image (image_filename_out).  This CUDA kernel performs
 * a simple 2D transform (rotation) on the texture coordinates (u,v).
 */

// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>  // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>  // helper functions for CUDA error check

#define MAX_EPSILON_ERROR 5e-3f

// Define the files that are to be save and the reference images for validation
const char *imageFilename = "teapot512.pgm";
const char *refFilename = "ref_rotated.pgm";

// Declaration, forward
void testFilters(int argc, char **argv);

// Constants for convolution
#define MAX_KERNEL_SIZE 32
#define BLOCK_SIZE 16

// Structure to hold filter information
struct ConvolutionFilter {
    float kernel[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
    int kernelSize;
};

// New convolution kernel
__global__ void convolutionKernel(float *outputData, int width, int height,
                                  cudaTextureObject_t tex, ConvolutionFilter filter) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int filterRadius = filter.kernelSize / 2;

    // Apply convolution
    for (int fy = -filterRadius; fy <= filterRadius; fy++) {
        for (int fx = -filterRadius; fx <= filterRadius; fx++) {
            float u = (float) (x + fx) / (float) width;
            float v = (float) (y + fy) / (float) height;

            float pixelValue = tex2D<float>(tex, u, v);
            float filterValue = filter.kernel[fy + filterRadius][fx + filterRadius];

            sum += pixelValue * filterValue;
        }
    }

    outputData[y * width + x] = sum;
}

__global__ void convolutionSharedKernel(float *outputData, int width, int height,
                                        cudaTextureObject_t tex, ConvolutionFilter filter) {
    // Shared memory for the image tile
    __shared__ float sharedMem[BLOCK_SIZE + MAX_KERNEL_SIZE - 1][BLOCK_SIZE + MAX_KERNEL_SIZE - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int x = bx + tx;
    int y = by + ty;

    int filterRadius = filter.kernelSize / 2;

    // Load the main pixel block into shared memory
    if (x < width && y < height) {
        float u = (float) x / (float) width;
        float v = (float) y / (float) height;
        sharedMem[ty][tx] = tex2D<float>(tex, u, v);
    }

    // Load halo region
    if (tx < filterRadius) {
        // Left halo
        if (x >= filterRadius && y < height) {
            float u = (float) (x - filterRadius) / (float) width;
            float v = (float) y / (float) height;
            sharedMem[ty][tx - filterRadius] = tex2D<float>(tex, u, v);
        }
        // Right halo
        if (x + BLOCK_SIZE < width && y < height) {
            float u = (float) (x + BLOCK_SIZE) / (float) width;
            float v = (float) y / (float) height;
            sharedMem[ty][tx + BLOCK_SIZE] = tex2D<float>(tex, u, v);
        }
    }

    if (ty < filterRadius) {
        // Top halo
        if (y >= filterRadius && x < width) {
            float u = (float) x / (float) width;
            float v = (float) (y - filterRadius) / (float) height;
            sharedMem[ty - filterRadius][tx] = tex2D<float>(tex, u, v);
        }
        // Bottom halo
        if (y + BLOCK_SIZE < height && x < width) {
            float u = (float) x / (float) width;
            float v = (float) (y + BLOCK_SIZE) / (float) height;
            sharedMem[ty + BLOCK_SIZE][tx] = tex2D<float>(tex, u, v);
        }
    }

    __syncthreads();

    // Compute convolution
    if (x < width && y < height) {
        float sum = 0.0f;

        for (int fy = -filterRadius; fy <= filterRadius; fy++) {
            for (int fx = -filterRadius; fx <= filterRadius; fx++) {
                float pixelValue = sharedMem[ty + fy + filterRadius][tx + fx + filterRadius];
                float filterValue = filter.kernel[fy + filterRadius][fx + filterRadius];
                sum += pixelValue * filterValue;
            }
        }

        outputData[y * width + x] = sum;
    }
}

// Function to create some common filters
ConvolutionFilter createFilter(const char *filterType) {
    ConvolutionFilter filter;

    if (strcmp(filterType, "emboss") == 0) {
        filter.kernelSize = 5;
        for (int i = 0; i < filter.kernelSize; i++)
            for (int j = 0; j < filter.kernelSize; j++)
                filter.kernel[i][i] = 0.0f;
        filter.kernel[0][0] = 1.0f;
        filter.kernel[1][1] = 1.0f;
        filter.kernel[3][3] = -1.0f;
        filter.kernel[4][4] = -1.0f;
    } else if (strcmp(filterType, "sharpen") == 0) {
        filter.kernelSize = 3;
        for (int i = 0; i < filter.kernelSize; i++)
            for (int j = 0; j < filter.kernelSize; j++)
                filter.kernel[i][j] = -1.0f;
        filter.kernel[1][1] = 9.0f;
    } else if (strcmp(filterType, "average") == 0) {
        filter.kernelSize = 5;
        float num = 1.0f / (filter.kernelSize * filter.kernelSize);
        for (int i = 0; i < filter.kernelSize; i++)
            for (int j = 0; j < filter.kernelSize; j++)
                filter.kernel[i][j] = num;
    }

    return filter;
}

void sequentialConvolution(float *input, float *output, int width, int height,
                           float *filter, int filterSize) {
    int filterRadius = filterSize / 2;

    // For each pixel in the image
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;

            // Apply filter
            for (int fy = -filterRadius; fy <= filterRadius; fy++) {
                for (int fx = -filterRadius; fx <= filterRadius; fx++) {
                    // Calculate source pixel position with clamping at borders
                    int srcX = min(max(x + fx, 0), width - 1);
                    int srcY = min(max(y + fy, 0), height - 1);

                    // Get pixel value and corresponding filter value
                    float pixelValue = input[srcY * width + srcX];
                    float filterValue = filter[(fy + filterRadius) * filterSize +
                                               (fx + filterRadius)];

                    sum += pixelValue * filterValue;
                }
            }

            // Store result
            output[y * width + x] = sum;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    printf("%s starting...\n");

    testFilters(argc, argv);

    printf("%s completed, returned %s\n");


    exit(EXIT_SUCCESS);
}

void testFilters(int argc, char **argv) {
    int devID = findCudaDevice(argc, (const char **) argv);

    // load image from disk
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

    if (imagePath == NULL) {
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);

    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

    // Allocate device memory for result
    float *dData = NULL;
    checkCudaErrors(cudaMalloc((void **)&dData, size));

    // Allocate array and copy image data
    cudaChannelFormatDesc channelDesc =
            cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cuArray;
    checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, width, height));
    checkCudaErrors(
        cudaMemcpyToArray(cuArray, 0, 0, hData, size, cudaMemcpyHostToDevice));

    cudaTextureObject_t tex;
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = cuArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));

    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    ConvolutionFilter filter = createFilter("blur");

    // Warmup
    convolutionKernel<<<dimGrid, dimBlock, 0>>>(dData, width, height, tex, filter);

    checkCudaErrors(cudaDeviceSynchronize());
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Execute the kernel
    convolutionKernel<<<dimGrid, dimBlock, 0>>>(dData, width, height, tex, filter);

    // dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
    //               (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    //
    // convolutionSharedKernel<<<gridSize, blockSize>>>(dData, width, height, tex, filter);

    // Check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n",
           (width * height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&timer);

    // Allocate mem for the result on host side
    float *hOutputData = (float *) malloc(size);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(hOutputData, dData, size, cudaMemcpyDeviceToHost));

    // Write result to file
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_out.pgm");
    sdkSavePGM(outputFilename, hOutputData, width, height);
    printf("Wrote '%s'\n", outputFilename);

    checkCudaErrors(cudaDestroyTextureObject(tex));
    checkCudaErrors(cudaFree(dData));
    checkCudaErrors(cudaFreeArray(cuArray));
    free(imagePath);
}
