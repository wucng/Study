// https://blog.csdn.net/qq_41248872/article/details/93883978
// g++ hello.cpp `pkg-config opencv --cflags --libs`
/*
算法思路：

       一、进行图像角度纠正

       二、取出ROI区域，去掉多余的白边
	   
*/

/*
=======图像旋转+切边=======
*/
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
 
using namespace cv;
using namespace std;
 
Mat Check_Skew(Mat&);
void FindROI(Mat&);
 
int threshold_value = 100;
int max_level = 255;
const char* output_win = "Contours Result";
const char* roi_win = "Final Result";
 
int main(int argc, char** argv) {
	Mat src = imread("rotate.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	//namedWindow("input image", CV_WINDOW_AUTOSIZE);
	//imshow("input image", src);
	//namedWindow(output_win, CV_WINDOW_AUTOSIZE);
	Mat img_skew = Check_Skew(src);
	// namedWindow(roi_win, CV_WINDOW_AUTOSIZE);
	//createTrackbar("Threshold:", output_win, &threshold_value, max_level, FindROI);
	FindROI(img_skew);
 
	waitKey(0);
	return 0;
}
 
//角度矫正
Mat Check_Skew(Mat& src) {
	Mat gray_src,canny_output;
	cvtColor(src, gray_src, COLOR_BGR2GRAY);
 
	//边缘检测
	Canny(gray_src, canny_output, threshold_value, threshold_value * 2, 3, false);
 
	//轮廓查找
	vector<vector<Point>> contours;
	vector<Vec4i> hireachy;
	findContours(canny_output, contours, hireachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	Mat drawImg = Mat::zeros(src.size(), CV_8UC3);
	float maxw = 0;
	float maxh = 0;
	double degree = 0;
	//角度获取
	for (size_t t = 0; t < contours.size(); t++) {
		RotatedRect minRect = minAreaRect(contours[t]);
		degree = abs(minRect.angle);
		if (degree > 0) {
			maxw = max(maxw, minRect.size.width);
			maxh = max(maxh, minRect.size.height);
		}
	}
 
	//轮廓绘制
	RNG rng(12345);
	for (size_t t = 0; t < contours.size(); t++) {
		RotatedRect minRect = minAreaRect(contours[t]);
		if (maxw == minRect.size.width && maxh == minRect.size.height) {
			degree = minRect.angle;
			Point2f pts[4];
			minRect.points(pts);
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			for (int i = 0; i < 4; i++) {
				line(drawImg, pts[i], pts[(i + 1) % 4], color, 2, 8, 0);
			}
		}
	}
	printf("max contours width : %f\n", maxw);
	printf("max contours height : %f\n", maxh);
	printf("max contours angle : %f\n", degree);
	//imshow(output_win, drawImg);
 
	//获得旋转矩阵
	Point2f center(src.cols / 2, src.rows / 2);
	Mat rotm = getRotationMatrix2D(center, degree, 1.0);  
 
	//旋转图像
	Mat dst;
	warpAffine(src, dst, rotm, src.size(), INTER_LINEAR, 0, Scalar(255, 255, 255));  
 
	//显示结果
	//imshow("Correct Image", dst);
	imwrite("result.jpg", dst);
 
	return dst;
}
 
//去边
void FindROI(Mat& img) {
	Mat gray_src;
	cvtColor(img, gray_src, COLOR_BGR2GRAY);
	
	//边缘检测
	Mat canny_output;
	Canny(gray_src, canny_output, threshold_value, threshold_value * 2, 3, false);
 
	//轮廓查找
	vector<vector<Point>> contours;
	vector<Vec4i> hireachy;
	findContours(canny_output, contours, hireachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
 
	//绘制轮廓
	int minw = img.cols*0.75;
	int minh = img.rows*0.75;
	RNG rng(12345);
	Mat drawImage = Mat::zeros(img.size(), CV_8UC3);
	Rect bbox;
	for (size_t t = 0; t < contours.size(); t++) {
		//查找可倾斜的最小外接矩
		RotatedRect minRect = minAreaRect(contours[t]);
		//获得倾斜角度
		float degree = abs(minRect.angle);
		if (minRect.size.width > minw && minRect.size.height > minh && minRect.size.width < (img.cols - 5)) {
			printf("current angle : %f\n", degree);
			Point2f pts[4];
			minRect.points(pts);
			bbox = minRect.boundingRect();
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			for (int i = 0; i < 4; i++) {
				line(drawImage, pts[i], pts[(i + 1) % 4], color, 2, 8, 0);
			}
		}
	}
	//imshow(output_win, drawImage);
 
	//提取ROI区域
	if (bbox.width > 0 && bbox.height > 0) {
		Mat roiImg = img(bbox);
		//imshow(roi_win, roiImg);
	}
	return;
}