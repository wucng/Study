/**
* 1.使用纹理内存
* 2.加载图片(没有使用opencv)
* 3.实现图片任意旋转
*/
#include "common.h"
#include <helper_timer.h> 

// typedef float FLOAT;
// Define the files that are to be save and the reference images for validation
const char *imageFilename = "lena_bw.pgm";
const char *refFilename   = "ref_rotated.pgm";

const char *sampleName = "simpleTexture";

/**********************************************************/
// Constants
const float angle = M_PI/4; //0.5f;        // angle to rotate image by (in radians)

// Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex;

// Auto-Verification Code
bool testResult = true;


////////////////////////////////////////////////////////////////////////////////
//! Transform an image using texture lookups
//! @param outputData  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void transformKernel(float *outputData,
                                int width,
                                int height,
                                float theta)
{
    // calculate normalized texture coordinates
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float u = (float)x - (float)width/2; 
    float v = (float)y - (float)height/2; 
    float tu = u*cosf(theta) - v*sinf(theta); 
    float tv = v*cosf(theta) + u*sinf(theta); 

    tu /= (float)width; 
    tv /= (float)height; 

    // read from texture and write to global memory
    outputData[y*width + x] = tex2D(tex, tu+0.5f, tv+0.5f);
}

int main(int argc,char *argv[])
{   
    /**获取GPU的数量*/
    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    mycout<<"GPU numbers:"<<deviceCount<<endl;
    if(deviceCount<1)
    {
        mycout<<"没有可用的GPU设备"<<endl;
        exit(-1);
    }

    // 设置使用哪块GPU
    int devID=0;
    if(argc>1)
    {
        cudaDeviceProp deviceProps;
        devID = findCudaDevice(argc, (const char **) argv);
        // get number of SMs on this GPU
        checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
        printf("CUDA device [%s] has %d Multi-Processors\n", deviceProps.name, deviceProps.multiProcessorCount);
    }
        
    checkCudaErrors(cudaSetDevice(devID));


    // load image from disk
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }
    sdkLoadPGM(imagePath, &hData, &width, &height); // 加载数据

    unsigned int size = width * height * sizeof(float); // 图像总内存大小

    printf("Loaded '%s', %d x %d pixels, Memory space: %.3f(M)\n", imageFilename, width, height,(float)size/1024/1024);


    //Load reference image from image (output)
    // float *hDataRef = (float *) malloc(size);
    // char *refPath = sdkFindFilePath(refFilename, argv[0]);

    // if (refPath == NULL)
    // {
    //     printf("Unable to find reference image file: %s\n", refFilename);
    //     exit(EXIT_FAILURE);
    // }

    // sdkLoadPGM(refPath, &hDataRef, &width, &height);


    // Allocate device memory for result
    float *dData = NULL;
    checkCudaErrors(cudaMalloc((void **) &dData, size));

    // Allocate array and copy image data
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cuArray;
    checkCudaErrors(cudaMallocArray(&cuArray,
                                    &channelDesc,
                                    width,
                                    height));
    checkCudaErrors(cudaMemcpyToArray(cuArray,
                                      0,
                                      0,
                                      hData,
                                      size,
                                      cudaMemcpyHostToDevice));


    // Set texture parameters
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = true;    // access with normalized texture coordinates

    // Bind the array to the texture
    checkCudaErrors(cudaBindTextureToArray(tex, cuArray, channelDesc));

    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid((width+dimBlock.x-1) / dimBlock.x, (height+dimBlock.y-1)/ dimBlock.y, 1);
    // Warmup(程序预热)
    transformKernel<<<dimGrid, dimBlock, 0>>>(dData, width, height, angle);

    checkCudaErrors(cudaDeviceSynchronize());
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    
    // Execute the kernel（正式执行）
    transformKernel<<<dimGrid, dimBlock, 0>>>(dData, width, height, angle);

    // Check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n",
           (width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&timer);

    // Allocate mem for the result on host side
    float *hOutputData = (float *) malloc(size);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(hOutputData,
                               dData,
                               size,
                               cudaMemcpyDeviceToHost));
    
    // Write result to file
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_out.pgm");
    sdkSavePGM(outputFilename, hOutputData, width, height);
    printf("Wrote '%s'\n", outputFilename);



    checkCudaErrors(cudaFree(dData));
    checkCudaErrors(cudaFreeArray(cuArray));
    free(imagePath);
    // free(refPath);
    return 0;
}