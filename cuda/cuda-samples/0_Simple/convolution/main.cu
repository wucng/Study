// nvcc main.cu  -I ../../common/  -std=c++11

#include "convolution.cuh"

template <typename T>
struct MyData
{
    int img_height=0,img_width=0,w_h=0,w_w=0,out_h=0,out_w=0;
    T *img=NULL,*weight=NULL,*output=NULL;
    MyData():img_height(0),img_width(0),w_h(0),w_w(0),out_h(0),out_w(0),img(NULL),weight(NULL),output(NULL){};

    ~MyData()
    {
        checkError(cudaFree(img));
        checkError(cudaFree(weight));
        checkError(cudaFree(output));
    }

    void allocate(T *h_weight,int img_height,int img_width,int w_h,int w_w,int out_h,int out_w)
    {
        this->img_height=img_height;
        this->img_width=img_width;
        this->w_h=w_h;
        this->w_w=w_w;
        this->out_h=out_h;
        this->out_w=out_w;

        checkError(cudaMallocManaged((void **)&img, sizeof(T)*img_height*img_width));
        checkError(cudaMallocManaged((void **)&weight, sizeof(T)*w_h*w_w));
        checkError(cudaMallocManaged((void **)&output, sizeof(T)*out_h*out_w));

        // checkError(cudaDeviceSynchronize()); // 同步,CPU等待GPU全部完成

        for (int i=0; i<img_height*img_width; i++)
        {
            img[i] = 1; //(T)(rand()%3);
        }

        for(int i=0; i<w_h*w_w; i++)
        {
            weight[i]=h_weight[i];
        }

        for (int i=0; i<out_h*out_w; i++)
            output[i] = 0;

    }

    void pprint()
    {
        mycout<<"img:"<<endl;
        for (int i=0; i<img_height; i++)
        {
            for (int j=0;j<img_width;++j)
            {
                cout<<img[j+i*img_width]<<" ";
            }
            cout<<endl;
        }

        mycout<<"output:"<<endl;
        for (int i=0; i<out_h; i++)
        {
            for (int j=0;j<out_w;++j)
            {
                cout<<output[j+i*out_w]<<" ";
            }
            cout<<endl;
        }
    }
};

int main()
{
    // 计算时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int img_height=32,img_width=32,w_h=3,w_w=3,out_h=32,out_w=32;
    // img_height=img_width=out_h=out_w=128;

    FLOAT h_weight[]={0,1,0,1,-5,1,0,1,0};
    MyData<FLOAT> data;
    data.allocate(h_weight,img_height,img_width,w_h,w_w,out_h,out_w);

    #if 0
    {
        dim3 threads(512,1,1);
        dim3 grid((img_height*img_width+threads.x-1) /threads.x,1,1);

        // Time copies and kernel
        cudaEventRecord(start,0);
        global_conv<FLOAT><<<grid,threads>>>(data.img,data.output,data.weight,data.img_height,data.img_width,data.w_h);//0.2968 (ms)
    }
    #else
    {
        dim3 threads(w_h*w_w,1,1);
        dim3 grid(out_h*out_w,1,1);

        // Time copies and kernel
        cudaEventRecord(start,0);
        shared_conv<9><<<grid,threads>>>(data.img,data.output,data.weight,data.img_height,data.img_width,data.w_h);//0.59184 (ms)
    }
    #endif
    
    // checkCudaErrors(cudaGetLastError());// launch vectorAdd kernel
    checkError(cudaDeviceSynchronize()); // 同步,CPU等待GPU全部完成
    // data.pprint();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop);

    mycout<<kernel_time<<" (ms)"<<endl;

    return 0;
}


int main01()
{
    // 计算时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int img_height=32,img_width=32,w_h=3,w_w=3,out_h=32,out_w=32;
    img_height=img_width=out_h=out_w=128;

    FLOAT h_weight[]={0,1,0,1,-5,1,0,1,0};
    MyData<FLOAT> data;
    data.allocate(h_weight,img_height,img_width,w_h,w_w,out_h,out_w);

    dim3 threads(w_h*w_w,32,1);
    dim3 grid(out_h*out_w/32,32,1);

    // Time copies and kernel
    cudaEventRecord(start,0);
    shared_conv2<32,9><<<grid,threads>>>(data.img,data.output,data.weight,data.img_height,data.img_width,data.w_h);//1.56534 (ms)

    // checkCudaErrors(cudaGetLastError());// launch vectorAdd kernel
    checkError(cudaDeviceSynchronize()); // 同步,CPU等待GPU全部完成
    // data.pprint();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop);

    mycout<<kernel_time<<" (ms)"<<endl;

    return 0;
}