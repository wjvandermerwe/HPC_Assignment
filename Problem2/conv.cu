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
struct FilterSettings {
    float kernel[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
    int kernelSize;
};

struct ConvolutionFilter {
    float* d_filter; // destination memory
    float* filterArray; // 1d input array
    int filterBytes; // size
    FilterSettings settings;
};

// Function to create some common filters
ConvolutionFilter createFilter(const char *filterType) {
    FilterSettings filter;
    ConvolutionFilter response;

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

    response.filterBytes = filter.kernelSize * filter.kernelSize * sizeof(float);

    checkCudaErrors(cudaMalloc(&response.d_filter, response.filterBytes));

    response.filterArray = new float[filter.kernelSize * filter.kernelSize];

    // Add bounds check for filter.kernel
    if (filter.kernel == nullptr) {
        fprintf(stderr, "Error: filter.kernel is null!\n");
        fflush(stderr);
        exit(1);
    }

    for(int i = 0; i < filter.kernelSize; i++) {
        for(int j = 0; j < filter.kernelSize; j++) {
            response.filterArray[i * filter.kernelSize + j] = filter.kernel[i][j];
        }
    }
    response.settings = filter;
    return response;
}

// New convolution kernel
// 1d array filter and incomming image
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
            // x + fx normalised texel values
            // x being current thread = pixel
            // fx being threshold value withing filter around current pixel
            float u = (float) (x + fx) / (float) width;
            float v = (float) (y + fy) / (float) height;

            float pixelValue = tex2D<float>(tex, u, v);
            // fy = -1 and fx =
            float filterValue = filter[(fy + filterRadius) * filterSize + (fx + filterRadius)];

            sum += pixelValue * filterValue;
        }
    }

    outputData[y * width + x] = sum;
}

__global__ void convolutionSharedKernel(float *outputData, int width, int height,
                                       cudaTextureObject_t tex, const float* filter, 
                                       int filterSize) {
    extern __shared__ float sharedMem[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int x = bx + tx;
    int y = by + ty;

    // Calculate shared memory dimensions
    const int sharedDim = BLOCK_SIZE + filterSize - 1;
    const int filterRadius = filterSize / 2;

    // Bounds checking helper
    auto inBounds = [width, height](int x, int y) {
        return x >= 0 && x < width && y >= 0 && y < height;
    };

    // Initialize shared memory to 0
    for (int i = ty; i < sharedDim; i += BLOCK_SIZE) {
        for (int j = tx; j < sharedDim; j += BLOCK_SIZE) {
            sharedMem[i * sharedDim + j] = 0.0f;
        }
    }
    __syncthreads();

    // Load the main block plus halo regions
    for (int dy = -filterRadius; dy <= filterRadius; dy++) {
        int sy = ty + dy + filterRadius;
        int gy = y + dy;
        
        for (int dx = -filterRadius; dx <= filterRadius; dx++) {
            int sx = tx + dx + filterRadius;
            int gx = x + dx;
            
            if (sx >= 0 && sx < sharedDim && sy >= 0 && sy < sharedDim) {
                if (inBounds(gx, gy)) {
                    float u = (float)gx / (float)width;
                    float v = (float)gy / (float)height;
                    sharedMem[sy * sharedDim + sx] = tex2D<float>(tex, u, v);
                }
            }
        }
    }
    __syncthreads();

    // Compute convolution only for valid output pixels
    if (x < width && y < height) {
        float sum = 0.0f;
        for (int fy = 0; fy < filterSize; fy++) {
            for (int fx = 0; fx < filterSize; fx++) {
                int sx = tx + fx;
                int sy = ty + fy;
                if (sx < sharedDim && sy < sharedDim) {
                    float pixelValue = sharedMem[sy * sharedDim + sx];
                    float filterValue = filter[fy * filterSize + fx];
                    sum += pixelValue * filterValue;
                }
            }
        }
        outputData[y * width + x] = sum;
    }
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



    cudaTextureObject_t tex = createTextureObject(cuArray);

    // allocate device memory for result
    float *hOutputRegular = NULL;
    float *hOutputShared = NULL;

    checkCudaErrors(cudaMalloc((void **)&hOutputRegular, size));
    checkCudaErrors(cudaMalloc((void **)&hOutputShared, size));

    printf("Debug: Starting convolution setup...\n");
    fflush(stdout);

    int blockSize = 16;
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(
        (width + blockSize - 1) / blockSize,
        (height + blockSize - 1) / blockSize
    );



    ConvolutionFilter filter = createFilter("average");


    checkCudaErrors(cudaMemcpy(filter.d_filter, filter.filterArray, filter.filterBytes, cudaMemcpyHostToDevice));


    // warmup to account for initialisation duration
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        fprintf(stderr, "[ERROR] Before launch: %s\n", cudaGetErrorString(error));
        fflush(stderr);
        exit(1);
    }

    convolutionKernel<<<dimGrid, dimBlock>>>(dData, width, height, tex, filter.d_filter, filter.settings.kernelSize);

    // Check for launch errors
    if(error != cudaSuccess) {
        fprintf(stderr, "[ERROR] Launch failed: %s\n", cudaGetErrorString(error));
        fflush(stderr);
        exit(1);
    }

    error = cudaDeviceSynchronize();
    if(error != cudaSuccess) {
        fprintf(stderr, "[ERROR] Execution failed: %s\n", cudaGetErrorString(error));
        fflush(stderr);
        exit(1);
    }

    size_t sharedMemSize = (BLOCK_SIZE + filter.settings.kernelSize - 1) *
                      (BLOCK_SIZE + filter.settings.kernelSize - 1) *
                      sizeof(float);

    printf("Debug: Shared memory size required: %zu bytes\n", sharedMemSize);
    fflush(stdout);

    convolutionSharedKernel<<<dimGrid, dimBlock, sharedMemSize>>>(
        dData, width, height, tex, filter.d_filter, filter.settings.kernelSize);

    if(error != cudaSuccess) {
        fprintf(stderr, "[ERROR] Launch failed: %s\n", cudaGetErrorString(error));
        fflush(stderr);
        exit(1);
    }

    error = cudaDeviceSynchronize();
    if(error != cudaSuccess) {
        fprintf(stderr, "[ERROR] Execution failed: %s\n", cudaGetErrorString(error));
        fflush(stderr);
        exit(1);
    }

    checkCudaErrors(cudaDeviceSynchronize());
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Execute the kernel
    // Regular CUDA convolution
    printf("\nTesting regular CUDA convolution:\n");
    sdkStartTimer(&timer);

    for(int i = 0; i < 10; i++) {
        convolutionKernel<<<dimGrid, dimBlock>>>(dData, width, height, tex, filter.d_filter, filter.settings.kernelSize);
    }
    getLastCudaError("Kernel execution failed");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    float regularTime = sdkGetTimerValue(&timer) / 10.0f;
    // float regularTime = sdkGetTimerValue(&timer);
    printf("Regular CUDA: %.2f ms (%.2f Mpixels/sec)\n",
           regularTime, (width * height / (regularTime / 1000.0f)) / 1e6);
    checkCudaErrors(cudaMemcpy(hOutputRegular, dData, size, cudaMemcpyDeviceToHost));
    sdkResetTimer(&timer);
    // Shared memory CUDA convolution
    printf("\nTesting shared memory CUDA convolution:\n");
    sdkStartTimer(&timer);

    for(int i = 0; i < 10; i++) {
        convolutionSharedKernel<<<dimGrid, dimBlock, sharedMemSize>>>(
            dData, width, height, tex, filter.d_filter, filter.settings.kernelSize);
    }
    getLastCudaError("Kernel execution failed");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    float sharedTime = sdkGetTimerValue(&timer) / 10.0f;
    // float sharedTime = sdkGetTimerValue(&timer);
    printf("Shared Memory CUDA: %.2f ms (%.2f Mpixels/sec)\n",
           sharedTime, (width * height / (sharedTime / 1000.0f)) / 1e6);
    checkCudaErrors(cudaMemcpy(hOutputShared, dData, size, cudaMemcpyDeviceToHost));
    sdkResetTimer(&timer);

    // Sequential CPU convolution
        printf("\nTesting sequential CPU convolution:\n");
    sdkStartTimer(&timer);

    // host memory
    float* hSequentialOutput = (float*)malloc(size);  // Use malloc for CPU memory
    if (!hSequentialOutput) {
        fprintf(stderr, "Failed to allocate host memory for sequential output\n");
        exit(1);
    }

    // Convert filter structure to array for CPU version
    sequentialConvolution(hData, hSequentialOutput, width, height, filter.filterArray, filter.settings.kernelSize);

    sdkStopTimer(&timer);
    float sequentialTime = sdkGetTimerValue(&timer);
    printf("Sequential CPU: %.2f ms (%.2f Mpixels/sec)\n",
           sequentialTime, (width * height / (sequentialTime / 1000.0f)) / 1e6);




    // Cleanup pointers
    sdkDeleteTimer(&timer);
    delete[] filter.filterArray;
    checkCudaErrors(cudaFree(filter.d_filter));
    checkCudaErrors(cudaFree(hOutputRegular));
    checkCudaErrors(cudaFree(hOutputShared));

    checkCudaErrors(cudaDestroyTextureObject(tex));
    checkCudaErrors(cudaFree(dData));
    checkCudaErrors(cudaFreeArray(cuArray));
    free(hSequentialOutput);
    free(imagePath);
}

cudaTextureObject_t createTextureObject(cudaArray *cuArray) {
    cudaTextureObject_t tex;
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = cuArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    // clamp to 0.0f
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.normalizedCoords = true;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));

    return tex;
}

void write_outputs(unsigned int size, float *dData, float *hOutputRegular, float *hOutputShared, char *imagePath) {
    // Allocate mem for the result on host side
    float *hOutputData = (float *) malloc(size);
    float *hOutputReg = (float *) malloc(size);
    float *hOutputSha = (float *) malloc(size);

    // copy result from device to host
    checkCudaErrors(cudaMemcpy(hOutputData, dData, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hOutputReg, hOutputRegular, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hOutputSha, hOutputShared, size, cudaMemcpyDeviceToHost));
    // Write result to file
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_out.pgm");
    char regOutputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_reg_out.pgm");
    char shaOutputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_sha_out.pgm");
    char seqOutputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_seq_out.pgm");
    sdkSavePGM(outputFilename, hOutputData, width, height);
    sdkSavePGM(regOutputFilename, hOutputReg, width, height);
    sdkSavePGM(shaOutputFilename, hOutputSha, width, height);
    sdkSavePGM(seqOutputFilename, hSequentialOutput, width, height);
    printf("Wrote '%s'\n", outputFilename);
}