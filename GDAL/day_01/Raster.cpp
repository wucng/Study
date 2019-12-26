// https://gdal.org/
// 编译 g++ xxx.cpp -lgdal
#include <iostream>
#include "gdal.h"
#include "gdal_priv.h"
#include "cpl_conv.h"
#include <string>
using namespace std;

int main() {
    // Opening the File
    GDALDataset *poDataset;
    GDALAllRegister();// 在打开GDAL支持的栅格数据存储之前，必须注册驱动程序
    // 导入gdal模块时，Python会自动调用GDALAllRegister（）。
    string pszFilename = "../test.jpg";
    poDataset = (GDALDataset *)GDALOpen(pszFilename.data(),GA_ReadOnly);// 传递数据集的名称和所需的访问权限（GA_ReadOnly或GA_Update）
    if(poDataset == NULL)
    {
    	cout << "open failed!" << endl; 
    }
	
	// Getting Dataset Information
	// 如栅格数据模型中所述，GDALDataset包含一系列栅格带，这些栅格带均属于同一区域且具有相同的分辨率。 
	// 它还具有元数据，坐标系，地理配准变换，栅格大小和各种其他信息。
	// 在特殊但常见的“北上”图像没有任何旋转或剪切的情况下，地理配准变换采用以下形式：
	// adfGeoTransform[0] /* top left x */
	// adfGeoTransform[1] /* w-e pixel resolution */
	// adfGeoTransform[2] /* 0 */
	// adfGeoTransform[3] /* top left y */
	// adfGeoTransform[4] /* 0 */
	// adfGeoTransform[5] /* n-s pixel resolution (negative value) */
	// 在一般情况下，这是仿射变换。
	
	double        adfGeoTransform[6];
	printf( "Driver: %s/%s\n",
			poDataset->GetDriver()->GetDescription(),
			poDataset->GetDriver()->GetMetadataItem( GDAL_DMD_LONGNAME ) );
	printf( "Size is %dx%dx%d\n",
			poDataset->GetRasterXSize(), poDataset->GetRasterYSize(),
			poDataset->GetRasterCount() );
	if( poDataset->GetProjectionRef()  != NULL )
		printf( "Projection is `%s'\n", poDataset->GetProjectionRef() );
	if( poDataset->GetGeoTransform( adfGeoTransform ) == CE_None )
	{
		printf( "Origin = (%.6f,%.6f)\n",
				adfGeoTransform[0], adfGeoTransform[3] );
		printf( "Pixel Size = (%.6f,%.6f)\n",
				adfGeoTransform[1], adfGeoTransform[5] );
	}
	
	// Fetching a Raster Band
	/*
	此时，通过GDAL一次只能访问一个波段。 此外，在每个频段上都有元数据，块大小，颜色表和其他各种可用信息。 
	以下代码从数据集中获取GDALRasterBand对象（编号为1到GDALRasterBand :: GetRasterCount（）），
	并显示有关该对象的一些信息。
	*/
	GDALRasterBand  *poBand;
	int             nBlockXSize, nBlockYSize;
	int             bGotMin, bGotMax;
	double          adfMinMax[2];
	poBand = poDataset->GetRasterBand( 1 );
	poBand->GetBlockSize( &nBlockXSize, &nBlockYSize );
	printf( "Block=%dx%d Type=%s, ColorInterp=%s\n",
			nBlockXSize, nBlockYSize,
			GDALGetDataTypeName(poBand->GetRasterDataType()),
			GDALGetColorInterpretationName(
				poBand->GetColorInterpretation()) );
	adfMinMax[0] = poBand->GetMinimum( &bGotMin );
	adfMinMax[1] = poBand->GetMaximum( &bGotMax );
	if( ! (bGotMin && bGotMax) )
		GDALComputeRasterMinMax((GDALRasterBandH)poBand, TRUE, adfMinMax);
	printf( "Min=%.3fd, Max=%.3f\n", adfMinMax[0], adfMinMax[1] );
	if( poBand->GetOverviewCount() > 0 )
		printf( "Band has %d overviews.\n", poBand->GetOverviewCount() );
	if( poBand->GetColorTable() != NULL )
		printf( "Band has a color table with %d entries.\n",
				poBand->GetColorTable()->GetColorEntryCount() );
	
	// Reading Raster Data
	/*
	有几种读取栅格数据的方法，但是最常见的方法是通过GDALRasterBand :: RasterIO（）方法。 
	此方法将自动处理数据类型转换，上/下采样和窗口化。 以下代码会将数据的第一条扫描线读取到大小相似的缓冲区中，
	并将其转换为浮点数作为操作的一部分。
	当不再使用pafScanline缓冲区时，应使用CPLFree（）释放它。
	*/
	float *pafScanline;
	int   nXSize = poBand->GetXSize();
	pafScanline = (float *) CPLMalloc(sizeof(float)*nXSize);
	CPLErr err= poBand->RasterIO( GF_Read, 0, 0, nXSize, 1,
					pafScanline, nXSize, 1, GDT_Float32,
					0, 0 ); // 读取第1行
	if (err!=0)
	{
		cout<<err<<endl<<"error"<<endl;
	}
	
	CPLFree(pafScanline);
	GDALClose(poDataset); // 关闭
    return 0;
}