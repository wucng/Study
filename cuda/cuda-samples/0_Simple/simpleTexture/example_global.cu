/**
* 1.使用全局内存
* 2.加载图片(没有使用opencv)
* 3.旋转后像素坐标会变成浮点数，(转成int)会导致目标像素坐标并不连续，而导致失败，
                            而使用纹理内存，可以直接操作浮点像素坐标
* 4.nvcc example_global.cu -I ../../common/ -std=c++11
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
const float angle = M_PI/2; //0.5f;        // angle to rotate image by (in radians)

// Auto-Verification Code
bool testResult = true;


////////////////////////////////////////////////////////////////////////////////
//! Transform an image using texture lookups
//! @param outputData  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void transformKernel(float *dData_in,float *dData_out,
                                int width,
                                int height,
                                float theta)
{
    // calculate normalized texture coordinates
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // float u = (float)x - (float)width/2; 
    // float v = (float)y - (float)height/2; 
    float u=(float)x;
    float v=(float)y;
    float tu = u*cosf(theta) - v*sinf(theta); 
    float tv = v*cosf(theta) + u*sinf(theta); 
    
    // tu /= (float)width; 
    // tv /= (float)height; 
    
    int position=(int)(tu+0.5+tv*width);

    // read from texture and write to global memory
    // outputData[y*width + x] = tex2D(tex, tu+0.5f, tv+0.5f);
    // dData_out[y*width + x] = dData_in[(int)(tv*width+tu)];
    if (position>=0 && position<width*height)
        dData_out[position] = dData_in[y*width + x];
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
        devID = findCudaDevice(argc, (const char **) argv);
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

    float *dData_in=NULL,*dData_out=NULL;
    checkCudaErrors(cudaMallocManaged((void **)&dData_out,size)); //使用统一虚拟内存，GPU与CPU都可以访问
    checkCudaErrors(cudaMalloc((void **)&dData_in,size));
    checkCudaErrors(cudaMemcpy(dData_in,hData,size,cudaMemcpyHostToDevice));

    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid((width+dimBlock.x-1) / dimBlock.x, (height+dimBlock.y-1)/ dimBlock.y, 1);
    // Warmup(程序预热)
    transformKernel<<<dimGrid, dimBlock, 0>>>(dData_in,dData_out,width, height, angle);

    checkCudaErrors(cudaDeviceSynchronize());
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    
    // Execute the kernel（正式执行）
    transformKernel<<<dimGrid, dimBlock, 0>>>(dData_in,dData_out,width, height, angle);

    // Check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n",
           (width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&timer);
    
    // Write result to file
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_out.pgm");
    sdkSavePGM(outputFilename, dData_out, width, height);
    printf("Wrote '%s'\n", outputFilename);


    // free
    checkCudaErrors(cudaFree(dData_in));
    checkCudaErrors(cudaFree(dData_out));
    free(imagePath);
    return 0;
}