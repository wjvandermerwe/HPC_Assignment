#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

void testFiltersOnImage(const char *filename);
cudaTextureObject_t createTextureObject(int width, int height, float *hData, unsigned int size);
void writeOutputs(unsigned int size, float *dData, float *hOutputRegular, float *hOutputShared, float *hSequentialOutput, char *imagePath, int height, int width);

// Constants
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

struct MethodTime {
    double totReg;
    double totSha;
    double totSeq;
};

struct Experiment {
    char* image_name;
    char* filter_name;
    MethodTime times;
};

struct Metrics {
    std::vector<Experiment> experiments;
    MethodTime total_times;
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
            // example fy = -1 and fx = -1
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
            output[y * width + x] = sum;
        }
    }
}

int main(int argc, char **argv) {
    printf("starting test\n");

    const char* files[] = {
        "image1.pgm",
        "image2.pgm",
        "image3.pgm"
    };
    int nFiles     = sizeof(files) / sizeof(files[0]);
    double totReg  = 0.0;
    double totSha  = 0.0;
    double totSeq  = 0.0;
    for(int idx = 0; idx < nFiles; ++idx) {
        testFiltersOnImage(files[idx]);
    }
    printf("test completed, returned\n");

    exit(EXIT_SUCCESS);
}

void testFiltersOnImage(const char *filename, double totReg, double totSha, double totSeq) {
    char imageFilename[1024];
    strcpy(imageFilename, filename);
    strcpy(imageFilename + strlen(filename) - 4, "_in.pgm");
    printf("Testing filters on image: %s\n", imageFilename);
    findCudaDevice(argc, (const char **) argv);

    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(filename, argv[0]);

    if (imagePath == NULL) {
        printf("Unable to source image file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);

    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

    const char* filters[] = {
        "emboss",
        "sharpen",
        "average"
    };
    int nFilts     = sizeof(filters) / sizeof(filters[0]);
    ConvolutionFilter filtersArr[3];
    for (int i = 0; i < nFilts; ++i) {
        filtersArr[i] = createFilter(filterNames[i]);
        checkCudaErrors(cudaMemcpy(
            filtersArr[i].d_filter,
            filtersArr[i].filterArray,
            filtersArr[i].filterBytes,
            cudaMemcpyHostToDevice
        ));
    }

    for (int idx = 0; idx < nFilts; ++idx) {
        ConvolutionFilter filter = filtersArr[idx];
        // image data
        float *dData = NULL;
        checkCudaErrors(cudaMalloc((void **)&dData, size));

        // move image into texture memory (spatial locality benefits)
        cudaTextureObject_t tex = createTextureObject(width, height, hData, size);
        // allocate device memory for result
        float *hOutputRegular = NULL; // device pointers
        float *hOutputShared = NULL;
        checkCudaErrors(cudaMalloc((void **)&hOutputRegular, size));
        checkCudaErrors(cudaMalloc((void **)&hOutputShared, size));



        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(
            (width + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (height + BLOCK_SIZE - 1) / BLOCK_SIZE
        );
        size_t sharedMemSize = (BLOCK_SIZE + filter.settings.kernelSize - 1) *
                          (BLOCK_SIZE + filter.settings.kernelSize - 1) *
                          sizeof(float);


        // WARMUP RUN (fills cache with intial config loads)
        convolutionKernel<<<dimGrid, dimBlock>>>(dData, width, height, tex, filter.d_filter, filter.settings.kernelSize);
        checkCudaErrors(cudaDeviceSynchronize());
        convolutionSharedKernel<<<dimGrid, dimBlock, sharedMemSize>>>(
            dData, width, height, tex, filter.d_filter, filter.settings.kernelSize);
        checkCudaErrors(cudaDeviceSynchronize());

        StopWatchInterface *timer = NULL;
        sdkCreateTimer(&timer);

        // Start tests





        // outputs
        writeOutputs(size, dData, hOutputRegular, hOutputShared, hSequentialOutput, imagePath, height, width);

        // Cleanup pointers
        sdkDeleteTimer(&timer);
        delete[] filter.filterArray;
        checkCudaErrors(cudaFree(filter.d_filter));
        checkCudaErrors(cudaFree(hOutputRegular));
        checkCudaErrors(cudaFree(hOutputShared));

        checkCudaErrors(cudaDestroyTextureObject(tex));
        checkCudaErrors(cudaFree(dData));
        // checkCudaErrors(cudaFreeArray(cuArray));
        free(hSequentialOutput);
    }
    free(imagePath);
}

struct KernelConfig {
    dim3 dimBlock;
    dim3 dimGrid;
    size_t sharedMemSize;
};

struct ExperimentConfig {
    unsigned int size;
    int width;
    int height;
    ConvolutionFilter filter;
    cudaTextureObject_t tex;
    float *d_data;
    float *output;
    StopWatchInterface *timer;
    KernelConfig kernelConfig;
};

float runConvKernel(ExperimentConfig params) {
    // start experiment for image
    // conv kernel first
    auto& [size, width, height, filter, tex, d_data, output, timer, kernelConfig] = params;

    printf("\nTesting regular CUDA convolution:\n");
    sdkStartTimer(&timer);
    for(int i = 0; i < 10; i++) {
        convolutionKernel<<<kernelConfig.dimGrid, kernelConfig.dimBlock>>>(d_data, width, height, tex, filter.d_filter, filter.settings.kernelSize);
        checkCudaErrors(cudaDeviceSynchronize());
    }
    sdkStopTimer(&timer);
    float regularTime = sdkGetTimerValue(&timer) / 10.0f;
    printf("Regular CUDA: %.2f ms (%.2f Mpixels/sec)\n",
           regularTime, (width * height / (regularTime / 1000.0f)) / 1e6);
    // copy output to host
    checkCudaErrors(cudaMemcpy(output, d_data, size, cudaMemcpyDeviceToHost));

    return regularTime;
}

double runConvKernelShared(int width, int height, unsigned int size, ConvolutionFilter filter, cudaTextureObject_t tex, float *dData, float *hOutputShared, StopWatchInterface *timer) {
    printf("\nTesting shared memory CUDA convolution:\n");
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    for(int i = 0; i < 10; i++) {
        convolutionSharedKernel<<<dimGrid, dimBlock, sharedMemSize>>>(
            dData, width, height, tex, filter.d_filter, filter.settings.kernelSize);
    }
    getLastCudaError("Kernel execution failed");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    float sharedTime = sdkGetTimerValue(&timer) / 10.0f;
    checkCudaErrors(cudaMemcpy(hOutputShared, dData, size, cudaMemcpyDeviceToHost));

    //Mpixels/sec
    return width * height / (sharedTime / 1000.0f) / 1e6;
}

float runSequentialConv(int width, int height, unsigned int size, ConvolutionFilter filter, float *hData, StopWatchInterface *timer) {
    // sequential cpu convolution
    printf("\nTesting sequential CPU convolution:\n");
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    // cpu memory
    float* hSequentialOutput = (float*)malloc(size);
    if (!hSequentialOutput) {
        fprintf(stderr, "Failed to allocate host memory for sequential output\n");
        exit(1);
    }

    sequentialConvolution(hData, hSequentialOutput, width, height, filter.filterArray, filter.settings.kernelSize);

    sdkStopTimer(&timer);
    float sequentialTime = sdkGetTimerValue(&timer);
    // printf("Sequential CPU: %.2f ms (%.2f Mpixels/sec)\n",
    //        sequentialTime, (width * height / (sequentialTime / 1000.0f)) / 1e6);

    return sequentialTime;
}


cudaTextureObject_t createTextureObject(int width, int height, float *hData, unsigned int size) {
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

void writeOutputs(unsigned int size, float *dData, float *hOutputRegular, float *hOutputShared, float *hSequentialOutput, char *imagePath, int height, int width) {

    float *hOutputData = (float *) malloc(size);
    float *hOutputReg = (float *) malloc(size);
    float *hOutputSha = (float *) malloc(size);

    checkCudaErrors(cudaMemcpy(hOutputData, dData, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hOutputReg, hOutputRegular, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hOutputSha, hOutputShared, size, cudaMemcpyDeviceToHost));

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