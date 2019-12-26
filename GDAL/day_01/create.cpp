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

	
	/*Using Create()*/
	const char *pszDstFilename = "../test2.jpg";
	GDALDataset *poDstDS;
	char **papszOptions = NULL; // 设置进度条 0...10...20...30...40...50...60...70...80...90...100 - done.
	papszOptions = CSLSetNameValue( papszOptions, "TILED", "YES" );
	papszOptions = CSLSetNameValue( papszOptions, "COMPRESS", "PACKBITS" );
	poDstDS = poDriver->Create( pszDstFilename, 512, 512, 1, GDT_Byte,
								papszOptions ); // 512x512x1
	
	double adfGeoTransform[6] = { 444720, 30, 0, 3751320, 0, -30 };
	OGRSpatialReference oSRS;
	char *pszSRS_WKT = NULL;
	GDALRasterBand *poBand;
	GByte abyRaster[512*512];
	poDstDS->SetGeoTransform( adfGeoTransform );
	oSRS.SetUTM( 11, TRUE );
	oSRS.SetWellKnownGeogCS( "NAD27" );
	oSRS.exportToWkt( &pszSRS_WKT );
	poDstDS->SetProjection( pszSRS_WKT );
	CPLFree( pszSRS_WKT );
	poBand = poDstDS->GetRasterBand(1);
	CPLErr err = poBand->RasterIO( GF_Write, 0, 0, 512, 512,
					abyRaster, 512, 512, GDT_Byte, 0, 0 );//GDT_Float32
	if (err!=0)
	{
		cout<<err<<endl<<"error"<<endl;
	}
	/* Once we're done, close properly the dataset */
	GDALClose( (GDALDatasetH) poDstDS );
}