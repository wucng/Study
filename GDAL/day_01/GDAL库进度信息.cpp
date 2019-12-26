// GDAL库进度信息编写示例
// https://blog.csdn.net/liminlu0314/article/details/51019220
// 编译 g++ xxx.cpp -lgdal

#include "gdal.h"
#include "gdal_priv.h"
#include "cpl_conv.h"

/**
* @brief 示例函数
* @param pszSrcFile         输入数据
* @param pszDstFie          输出数据
* @param pfnProgress        进度信息回调函数
* @param pProgressArg       进度信息回调函数参数
* @return                   返回值，处理成功返回CE_None
*/
CPLErr TestFunction(const char* pszSrcFile,
                    const char* pszDstFile,
                    GDALProgressFunc pfnProgress = NULL,
                    void *  pProgressArg = NULL)    
{
    // 如果没有指定进度条回调函数，使用GDAL库中的默认回调函数
    if(pfnProgress == NULL)
        pfnProgress = GDALDummyProgress;

    // 设置进度信息以及初值为0，可同时设置处理信息
    if(!pfnProgress(0.0, "Start ....", pProgressArg))
    {
        CPLError(CE_Failure, CPLE_UserInterrupt, "User terminated");
        return CE_Failure;
    }

    // 一个示例的循环，里面描述了更新进度信息
    for (int i=0; i<100; i++)
    {
        //do something

        if(!pfnProgress((i+1.0)/100, "Processing ....", pProgressArg))
        {
            CPLError(CE_Failure, CPLE_UserInterrupt, "User terminated");
            return CE_Failure;
        }
    }

    // 处理完成，将进度信息更新为1，可同时设置处理信息
    if(!pfnProgress(1.0, "End ....", pProgressArg))
    {
        // CPLError(CE_Failure, CPLErnterrupt, "User terminated");
		CPLError(CE_Failure, CPLE_UserInterrupt, "User terminated");
        return CE_Failure;
    }

    return CE_None;
}

int main()
{
    //进度条回调函数，这里使用GDAL自带的一个控制台进度函数
    GDALProgressFunc pfnProgress = GDALTermProgress;
     //进度条回调函数参数，该参数需与回调函数配合使用

    void *  pProgressArg = NULL;    
	const char *pszSrcFile = "input.tif";
    const char *pszDstFile = "Output.tif";

    CPLErr Err = TestFunction(pszSrcFile, pszDstFile, pfnProgress, pProgressArg);

    return 0;
}