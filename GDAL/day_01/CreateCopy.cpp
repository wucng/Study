#include "cpl_string.h"
#include <iostream>
#include "gdal.h"
#include "gdal_priv.h"
#include "cpl_conv.h"
#include <string>
using namespace std;

int main()
{
	GDALAllRegister();// 注册驱动
	// 查看支持CreateCopy，Create方法
	const char *pszFormat = "GTiff";
	GDALDriver *poDriver;
	char **papszMetadata;
    poDriver = GetGDALDriverManager()->GetDriverByName(pszFormat);
    if( poDriver == NULL )
        exit( 1 );
    papszMetadata = poDriver->GetMetadata();
    if( CSLFetchBoolean( papszMetadata, GDAL_DCAP_CREATE, FALSE ) )
        printf( "Driver %s supports Create() method.\n", pszFormat );
    if( CSLFetchBoolean( papszMetadata, GDAL_DCAP_CREATECOPY, FALSE ) )
        printf( "Driver %s supports CreateCopy() method.\n", pszFormat );
	//Driver GTiff supports Create() method.
	//Driver GTiff supports CreateCopy() method.

	
	/*Using CreateCopy()*/
	/*
	请注意，CreateCopy（）方法返回可写数据集，并且必须正确关闭它才能完成将数据集写入并刷新到磁盘的操作。 
	在Python情况下，当“ dst_ds”超出范围时会自动发生。 仅在CreateCopy（）调用中的目标文件名之后，
	用于bStrict选项的FALSE（或0）值指示即使无法创建目标数据集以使其与输入数据集完全匹配，
	CreateCopy（）调用也应继续进行而不会出现致命错误。 。 这可能是因为输出格式不支持输入数据集的像素数据类型，
	或者是因为目标不支持例如编写地理配准。
	*/
	const char *pszSrcFilename = "../test.jpg";
	const char *pszDstFilename = "../test2.jpg";
	
	GDALDataset *poSrcDS = // 源数据
	(GDALDataset *) GDALOpen( pszSrcFilename, GA_ReadOnly );
	GDALDataset *poDstDS;  // 目标数据
	/*
	poDstDS = poDriver->CreateCopy( pszDstFilename, poSrcDS, FALSE,
									NULL, NULL, NULL );
									
	// Once we're done, close properly the dataset//
	if( poDstDS != NULL )
		GDALClose( (GDALDatasetH) poDstDS );
	GDALClose( (GDALDatasetH) poSrcDS );
	*/
	
	// 更复杂的情况可能涉及传递创建选项，并使用预定义的进度监视器，如下所示：
	char **papszOptions = NULL; // 设置进度条 0...10...20...30...40...50...60...70...80...90...100 - done.
	papszOptions = CSLSetNameValue( papszOptions, "TILED", "YES" );
	papszOptions = CSLSetNameValue( papszOptions, "COMPRESS", "PACKBITS" );
	poDstDS = poDriver->CreateCopy( pszDstFilename, poSrcDS, FALSE,
									papszOptions, GDALTermProgress, NULL );
	/* Once we're done, close properly the dataset */
	if( poDstDS != NULL )
		GDALClose( (GDALDatasetH) poDstDS );
	CSLDestroy( papszOptions );
}