// GDAL库进度信息编写示例
// https://blog.csdn.net/liminlu0314/article/details/51019220

// GDAL分块处理简单的流程
// https://blog.csdn.net/liminlu0314/article/details/73881097

// 编译 g++ xxx.cpp -lgdal

/**
利用GDAL库函数创建图像时，一般会用到GDALDriver类Create()函数，但是Create()函数不支持JPEG、PNG等格式，
不过，CreateCopy()支持这些格式，所以根据已有的图像数据，不能直接创建jpg、png格式的图像，
而要借助GDAL的MEM内存文件，来创建他们。


除了利用MEM内存创建外，还可以借助C\C++的一些函数，
先用GDALDriver类的Create()函数生成.tif文件，
然后通过更改文件名，来实现把.tif格式转化为所需格式。

CString oldname = OutimgName; 
OutimgName.Replace(".tif",".jpg");//更改文件名后缀，改成.jpg格式
CString newname = OutimgName;
remove(newname);  //如果已有该jpg文件，则移除，以便产生新的，否则不能替换文件名
rename(oldname,newname);//替换文件名
*/

#include "gdal.h"
#include "gdal_priv.h"
#include "cpl_conv.h"

/**类型转换*/
unsigned char type_convers(double a)
{
    if (a < 0.0)
        a = 0.0;
    if (a > 255.0)
        a = 255.0;
    return static_cast<unsigned char>(a);
}

// 加上进度条
CPLErr ImageProcess(const char* pszSrcFile,
				    const char* pszDstFile, 
				    const char* pszFormat,
				    GDALProgressFunc pfnProgress = NULL,
                    void *  pProgressArg = NULL)
{
	
	// ***************设置进度********************************************//
	// 如果没有指定进度条回调函数，使用GDAL库中的默认回调函数
    if(pfnProgress == NULL)
        pfnProgress = GDALDummyProgress;
	
	// 设置进度信息以及初值为0，可同时设置处理信息
    if(!pfnProgress(0.0, "Start ....", pProgressArg))
    {
        CPLError(CE_Failure, CPLE_UserInterrupt, "User terminated");
        return CE_Failure;
    }
	// ***********************************************************//
	
    //注册GDAL驱动
    GDALAllRegister();

    //获取输出图像驱动
	GDALDriver *poDriver = GetGDALDriverManager()->GetDriverByName("MEM"); // Or GTiff	
	
    if (poDriver == NULL)   //输出文件格式错误
        // return false;
		return CE_Failure;

    //打开输入图像
    GDALDataset *poSrcDS = (GDALDataset*)GDALOpen(pszSrcFile, GA_ReadOnly);
    if (poSrcDS == NULL)    //输入文件打开失败
        // return false;
		return CE_Failure;

    //获取输入图像的宽高波段书
    int nXSize = poSrcDS->GetRasterXSize();
    int nYSize = poSrcDS->GetRasterYSize();
    int nBands = poSrcDS->GetRasterCount();

    //获取输入图像仿射变换参数
    double adfGeotransform[6] = { 0 };
    poSrcDS->GetGeoTransform(adfGeotransform);
    //获取输入图像空间参考
    const char* pszProj = poSrcDS->GetProjectionRef();

    GDALRasterBand *poBand = poSrcDS->GetRasterBand(1);
    if (poBand == NULL)    //获取输入文件中的波段失败
    {
        GDALClose((GDALDatasetH)poSrcDS);
        // return false;
		return CE_Failure;
    }

    //创建输出图像，输出图像是1个波段
	GDALDataset *poDstDS = poDriver->Create("", nXSize, nYSize, 1, GDT_Byte, NULL);
	
    if (poDstDS == NULL)    //创建输出文件失败
    {
        GDALClose((GDALDatasetH)poSrcDS);
        // return false;
		return CE_Failure;
    }

    //设置输出图像仿射变换参数，与原图一致
    poDstDS->SetGeoTransform(adfGeotransform);
    //设置输出图像空间参考，与原图一致
    poDstDS->SetProjection(pszProj);

    int nBlockSize = 256;     //分块大小

    //分配输入分块缓存
    unsigned char *pSrcData = new unsigned char[nBlockSize*nBlockSize*nBands];
    //分配输出分块缓存
    unsigned char *pDstData = new unsigned char[nBlockSize*nBlockSize];

    //定义读取输入图像波段顺序
    int *pBandMaps = new int[nBands];
    for (int b = 0; b < nBands; b++)
        pBandMaps[b] = b + 1;
	
	CPLErr Err;
    //循环分块并进行处理
    for (int i = 0; i < nYSize; i += nBlockSize)
    {
        for (int j = 0; j < nXSize; j += nBlockSize)
        {
            //定义两个变量来保存分块大小
            int nXBK = nBlockSize;
            int nYBK = nBlockSize;

            //如果最下面和最右边的块不够256，剩下多少读取多少
            if (i + nBlockSize > nYSize)     //最下面的剩余块
                nYBK = nYSize - i;
            if (j + nBlockSize > nXSize)     //最右侧的剩余块
                nXBK = nXSize - j;

            //读取原始图像块
            Err = poSrcDS->RasterIO(GF_Read, j, i, nXBK, nYBK, pSrcData, nXBK, nYBK, GDT_Byte, nBands, pBandMaps, 0, 0, 0, NULL);
			
            //再这里填写你自己的处理算法
            //pSrcData 就是读取到的分块数据，存储顺序为，先行后列，最后波段
            //pDstData 就是处理后的二值图数据，存储顺序为先行后列

            // memcpy(pDstData, pSrcData, sizeof(unsigned char)*nXBK*nYBK);
            //上面这句是一个测试，将原始图像的第一个波段数据复制到输出的图像里面
			
			for (int ii =0;ii < nYBK;++ii)
			{
				for (int jj = 0;jj < nXBK;++jj)
				{
					// opencv cuda numpy的存放方式 矩阵shape：nYBK*nXBK*nBands
					// pDstData[jj+ii*nXBK] = type_convers(0.299*pSrcData[(jj+ii*nXBK)*nBands+0]+0.587*pSrcData[(jj+ii*nXBK)*nBands+1]+0.114*pSrcData[(jj+ii*nXBK)*nBands+2]);				
					
					// 与上面的存放方式有差异  矩阵shape：nBands*nYBK*nXBK
					pDstData[jj+ii*nXBK] = type_convers(0.299*pSrcData[(jj+ii*nXBK)+0*nYBK*nXBK]+0.587*pSrcData[(jj+ii*nXBK)+1*nYBK*nXBK]+0.114*pSrcData[(jj+ii*nXBK)+2*nYBK*nXBK]);
				}
			}
			

            //写到结果图像
            Err = poDstDS->RasterIO(GF_Write, j, i, nXBK, nYBK, pDstData, nXBK, nYBK, GDT_Byte, 1, pBandMaps, 0, 0, 0, NULL);
        
			// ***************设置进度********************************************//
			if(!pfnProgress(((i+1.0)*nYBK+nXBK)/100, "Processing ....", pProgressArg))
			{
				CPLError(CE_Failure, CPLE_UserInterrupt, "User terminated");
				return CE_Failure;
			}
			// ***********************************************************//
		}
    }
	
	//以创建复制的方式，生成jpg文件
	GDALDriver *pDriverJPG = GetGDALDriverManager()->GetDriverByName(pszFormat);
	pDriverJPG->CreateCopy(pszDstFile,poDstDS,TRUE,0,0,0); //创建jpg文件
	
		
	
    //释放申请的内存
    delete[]pSrcData;
    delete[]pDstData;
    delete[]pBandMaps;

    //关闭原始图像和结果图像
    GDALClose((GDALDatasetH)poSrcDS);
    GDALClose((GDALDatasetH)poDstDS);
	
	// ***************设置进度********************************************//
	// 处理完成，将进度信息更新为1，可同时设置处理信息
    if(!pfnProgress(1.0, "End ....", pProgressArg))
    {
        // CPLError(CE_Failure, CPLErnterrupt, "User terminated");
		CPLError(CE_Failure, CPLE_UserInterrupt, "User terminated");
        return CE_Failure;
    }
	// ***********************************************************//
	
    return CE_None;
}

int main()
{
    //进度条回调函数，这里使用GDAL自带的一个控制台进度函数
    GDALProgressFunc pfnProgress = GDALTermProgress;
    //进度条回调函数参数，该参数需与回调函数配合使用

    void *  pProgressArg = NULL;
	const char* pszFormat = "JPEG"; // GTiff
	const char *pszSrcFile = "../test.jpg";
    const char *pszDstFile = "../test2.jpg";

    CPLErr Err = ImageProcess(pszSrcFile, pszDstFile, pszFormat, pfnProgress, pProgressArg);

    return 0;
}