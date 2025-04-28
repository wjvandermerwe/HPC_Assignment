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
                                  cudaTextureObject_t tex, const float* filter, int filterSize) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int filterRadius = filterSize / 2;

    // Apply convolution
    for (int fy = -filterRadius; fy <= filterRadius; fy++) {
        for (int fx = -filterRadius; fx <= filterRadius; fx++) {
            float u = (float) (x + fx) / (float) width;
            float v = (float) (y + fy) / (float) height;

            float pixelValue = tex2D<float>(tex, u, v);
            float filterValue = filter[(fy + filterRadius) * filterSize + (fx + filterRadius)];

            sum += pixelValue * filterValue;
        }
    }

    outputData[y * width + x] = sum;
}

__global__ void convolutionSharedKernel(float *outputData, int width, int height,
                                        cudaTextureObject_t tex, const float* filter, int filterSize) {
    // Shared memory for the image tile
    __shared__ float sharedMem[BLOCK_SIZE + MAX_KERNEL_SIZE - 1][BLOCK_SIZE + MAX_KERNEL_SIZE - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int x = bx + tx;
    int y = by + ty;

    int filterRadius = filterSize / 2;

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
                float filterValue = filter[(fy + filterRadius) * filterSize + (fx + filterRadius)];

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

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;

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
    printf("starting...\n");

    testFilters(argc, argv);

    printf("completed, returned\n");

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

    // Allocate device memory for result
    float *hOutputRegular = NULL;
    float *hOutputShared = NULL;
    float *hOutputSequential = NULL;
    checkCudaErrors(cudaMalloc((void **)&hOutputRegular, size));
    checkCudaErrors(cudaMalloc((void **)&hOutputShared, size));
    checkCudaErrors(cudaMalloc((void **)&hOutputSequential, size));

    int blockSize = 16; // Using a more standard block size
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(
        (width + blockSize - 1) / blockSize,
        (height + blockSize - 1) / blockSize
    );

    ConvolutionFilter filter = createFilter("blur");
    float* d_filter;
    int filterBytes = filter.kernelSize * filter.kernelSize * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_filter, filterBytes));
    float* filterArray = new float[filter.kernelSize * filter.kernelSize];
    for(int i = 0; i < filter.kernelSize; i++) {
        for(int j = 0; j < filter.kernelSize; j++) {
            filterArray[i * filter.kernelSize + j] = filter.kernel[i][j];
        }
    }
    checkCudaErrors(cudaMemcpy(d_filter, filterArray, filterBytes, cudaMemcpyHostToDevice));
    printf("Grid dimensions: %dx%d\n", dimGrid.x, dimGrid.y);
    printf("Block dimensions: %dx%d\n", dimBlock.x, dimBlock.y);
    printf("Memory allocation checks:\n");
    printf("Image dimensions: %d x %d\n", width, height);
    printf("Filter size: %d\n", filter.kernelSize);
    fprintf(stderr, "Checking device pointers:\n");
    fprintf(stderr, "dData: %p\n", dData);
    fprintf(stderr, "d_filter: %p\n", d_filter);
    fflush(stdout);
    fflush(stderr);
    // Warmup to account for initialisation duration
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        fprintf(stderr, "[ERROR] Before launch: %s\n", cudaGetErrorString(error));
        fflush(stderr);
        exit(1);  // Force exit with error code
    }


    convolutionKernel<<<dimGrid, dimBlock>>>(dData, width, height, tex, d_filter, filter.kernelSize);

    // Check for launch errors
    if(error != cudaSuccess) {
        fprintf(stderr, "[ERROR] Launch failed: %s\n", cudaGetErrorString(error));
        fflush(stderr);
        exit(1);  // Force exit with error code
    }

    error = cudaDeviceSynchronize();
    if(error != cudaSuccess) {
        fprintf(stderr, "[ERROR] Execution failed: %s\n", cudaGetErrorString(error));
        fflush(stderr);
        exit(1);  // Force exit with error code
    }


    // checkCudaErrors(cudaDeviceSynchronize());


    convolutionSharedKernel<<<dimGrid, dimBlock>>>(dData, width, height, tex, d_filter, filter.kernelSize);
    checkCudaErrors(cudaDeviceSynchronize());
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Execute the kernel
    // Regular CUDA convolution
    printf("\nTesting regular CUDA convolution:\n");
    sdkStartTimer(&timer);

    // for(int i = 0; i < 10; i++) {  // Run multiple iterations for better averaging
    convolutionKernel<<<dimGrid, dimBlock>>>(dData, width, height, tex, d_filter, filter.kernelSize);

    // }
    getLastCudaError("Kernel execution failed");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    // float regularTime = sdkGetTimerValue(&timer) / 10.0f;
    float regularTime = sdkGetTimerValue(&timer);
    printf("Regular CUDA: %.2f ms (%.2f Mpixels/sec)\n",
           regularTime, (width * height / (regularTime / 1000.0f)) / 1e6);
    checkCudaErrors(cudaMemcpy(hOutputRegular, dData, size, cudaMemcpyDeviceToHost));
    sdkResetTimer(&timer);
    // Shared memory CUDA convolution
    printf("\nTesting shared memory CUDA convolution:\n");
    sdkStartTimer(&timer);

    // for(int i = 0; i < 10; i++) {
    convolutionSharedKernel<<<dimGrid, dimBlock>>>(dData, width, height, tex, d_filter, filter.kernelSize);
    // }
    getLastCudaError("Kernel execution failed");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    // float sharedTime = sdkGetTimerValue(&timer) / 10.0f;
    float sharedTime = sdkGetTimerValue(&timer);
    printf("Shared Memory CUDA: %.2f ms (%.2f Mpixels/sec)\n",
           sharedTime, (width * height / (sharedTime / 1000.0f)) / 1e6);
    checkCudaErrors(cudaMemcpy(hOutputShared, dData, size, cudaMemcpyDeviceToHost));
    sdkResetTimer(&timer);
    // Sequential CPU convolution
        printf("\nTesting sequential CPU convolution:\n");
    sdkStartTimer(&timer);

    // Convert filter structure to array for CPU version
    sequentialConvolution(hData, hOutputSequential, width, height, filterArray, filter.kernelSize);

    sdkStopTimer(&timer);
    float sequentialTime = sdkGetTimerValue(&timer);
    printf("Sequential CPU: %.2f ms (%.2f Mpixels/sec)\n",
           sequentialTime, (width * height / (sequentialTime / 1000.0f)) / 1e6);

    // Allocate mem for the result on host side
    float *hOutputData = (float *) malloc(size);
    float *hOutputReg = (float *) malloc(size);
    float *hOutputSha = (float *) malloc(size);
    float *hOutputSeq = (float *) malloc(size);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(hOutputData, dData, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hOutputReg, hOutputRegular, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hOutputSha, hOutputShared, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hOutputSeq, hOutputSequential, size, cudaMemcpyDeviceToHost));
    // Write result to file
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_out.pgm");
    char regOutputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_out.pgm");
    char shaOutputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_out.pgm");
    char seqOutputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_out.pgm");
    sdkSavePGM(outputFilename, hOutputData, width, height);
    sdkSavePGM(regOutputFilename, hOutputReg, width, height);
    sdkSavePGM(shaOutputFilename, hOutputSha, width, height);
    sdkSavePGM(seqOutputFilename, hOutputSeq, width, height);
    printf("Wrote '%s'\n", outputFilename);


    // Cleanup
    sdkDeleteTimer(&timer);
    delete[] filterArray;
    checkCudaErrors(cudaFree(d_filter));
    checkCudaErrors(cudaFree(hOutputRegular));
    checkCudaErrors(cudaFree(hOutputShared));
    checkCudaErrors(cudaFree(hOutputSequential));

    checkCudaErrors(cudaDestroyTextureObject(tex));
    checkCudaErrors(cudaFree(dData));
    checkCudaErrors(cudaFreeArray(cuArray));
    free(imagePath);
}
