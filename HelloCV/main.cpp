#include <iostream>
#include "main.h"

using namespace std;
using namespace cv;

Mat g_img;
Point g_pointOld;

int main()
{
	/*int iNumber = GetNumber();

	Run(iNumber);*/

	Logical();
	
	return 0;
}

//int GetNumber()
//{
//	int iNumber;
//	cout << "1. MatOp\n2. Camera_in\n\n" << endl;
//	cout << "0. Exit" << endl;
//	cout << "Select Number: " << endl;
//	cin >> iNumber;
//	return iNumber;
//}

//void Run(int iNumber)
//{
//	switch (iNumber)
//	{
//	case 0:
//		return;
//	case 1:
//
//	case 2:
//	
//	default:
//
//	}
//}

void MatOp()
{
	Mat img1 = imread("lenna.bmp");

	cout << "Width: " << img1.cols << endl;
	cout << "Height: " << img1.rows << endl;
	cout << "Channels: " << img1.channels() << endl;

	if (img1.type() == CV_8UC1)
		cout << "img1 is a grayscale image." << endl;
	else
		cout << "img1 is a truecolor image." << endl;

	float data[] = { 2.f, 1.414f, 3.f, 1.732f };
	Mat mat1(2, 2, CV_32FC1, data);
	cout << "mat1:\n" << mat1 << endl;
}

void Camera_in()
{
	VideoCapture cam(0);

	if (!cam.isOpened())
	{
		cerr << "Camera open failed!" << endl;
		return;
	}

	cout << "Frame width: " << cvRound(cam.get(CAP_PROP_FRAME_WIDTH)) << endl;
	cout << "Frame height: " << cvRound(cam.get(CAP_PROP_FRAME_HEIGHT)) << endl;

	Mat frame;

	while (true)
	{
		cam >> frame;
		if (frame.empty())
			break;

		imshow("frame", frame);

		if (waitKey(10) == 27)	// ESC Key
			break;
	}
}

void VideoIn()
{
	VideoCapture cap("stopwatch.avi");

	if (!cap.isOpened()) {
		cerr << "Video open failed!" << endl;
		return;
	}

	cout << "Frame width: " << cvRound(cap.get(CAP_PROP_FRAME_WIDTH)) << endl;
	cout << "Frame height: " << cvRound(cap.get(CAP_PROP_FRAME_HEIGHT)) << endl;
	cout << "Frame count: " << cvRound(cap.get(CAP_PROP_FRAME_COUNT)) << endl;

	double fps = cap.get(CAP_PROP_FPS);
	cout << "FPS: " << fps << endl;

	//int delay = cvRound(1000 / fps) + 1;
	double delay = 1000 / fps;

	Mat frame;
	while (true) {
		cap >> frame;
		if (frame.empty())
			break;

		imshow("frame", frame);

		if (waitKey(delay) == 27) // ESC key
			break;
	}

	destroyAllWindows();
}

void DrawLines()
{
	Mat img(400, 400, CV_8UC3, Scalar(255, 255, 255));

	line(img, Point(50, 50), Point(200, 50), Scalar(0, 0, 255));
	line(img, Point(50, 100), Point(200, 100), Scalar(255, 0, 255), 3);
	line(img, Point(50, 150), Point(200, 150), Scalar(255, 0, 0), 10);

	line(img, Point(250, 50), Point(350, 100), Scalar(0, 0, 255), 1, LINE_4);
	line(img, Point(250, 70), Point(350, 120), Scalar(255, 0, 255), 1, LINE_8);
	line(img, Point(250, 90), Point(350, 140), Scalar(255, 0, 0), 1, LINE_AA);

	arrowedLine(img, Point(50, 200), Point(150, 200), Scalar(0, 0, 255), 1);
	arrowedLine(img, Point(50, 250), Point(350, 250), Scalar(255, 0, 255), 1);
	arrowedLine(img, Point(50, 300), Point(350, 300), Scalar(255, 0, 0), 1, LINE_8, 0, 0.05);

	drawMarker(img, Point(50, 350), Scalar(0, 0, 255), MARKER_CROSS);
	drawMarker(img, Point(100, 350), Scalar(0, 0, 255), MARKER_TILTED_CROSS);
	drawMarker(img, Point(150, 350), Scalar(0, 0, 255), MARKER_STAR);
	drawMarker(img, Point(200, 350), Scalar(0, 0, 255), MARKER_DIAMOND);
	drawMarker(img, Point(250, 350), Scalar(0, 0, 255), MARKER_SQUARE);
	drawMarker(img, Point(300, 350), Scalar(0, 0, 255), MARKER_TRIANGLE_UP);
	drawMarker(img, Point(350, 350), Scalar(0, 0, 255), MARKER_TRIANGLE_DOWN);

	imshow("img", img);
	waitKey();

	destroyAllWindows();
}

void DrawText()
{
	Mat img(200, 640, CV_8UC3, Scalar(255, 255, 255));

	const String text = "Hello, OpenCV";
	int fontFace = FONT_HERSHEY_TRIPLEX;
	double fontScale = 2.0;
	int thickness = 1;

	Size sizeText = getTextSize(text, fontFace, fontScale, thickness, 0);
	Size sizeImg = img.size();

	Point org((sizeImg.width - sizeText.width) / 2, (sizeImg.height + sizeText.height) / 2);
	putText(img, text, org, fontFace, fontScale, Scalar(255, 0, 0), thickness);
	rectangle(img, org, org + Point(sizeText.width, -sizeText.height), Scalar(0, 255, 0), 1);

	imshow("img", img);
	waitKey();

	destroyAllWindows();
}

void Keyboard()
{
	Mat img = imread("lenna.bmp");

	if (img.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	namedWindow("img");
	imshow("img", img);

	while (true) {
		int keycode = waitKey();

		if (keycode == 'i' || keycode == 'I') {
			img = ~img;
			imshow("img", img);
		}
		else if (keycode == 27 || keycode == 'q' || keycode == 'Q') {
			break;
		}
	}
}

void Mouse()
{
	g_img = imread("lenna.bmp");

	if (g_img.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	namedWindow("g_img");
	setMouseCallback("g_img", OnMouse);

	imshow("g_img", g_img);
	waitKey();
}

void OnMouse(int event, int x, int y, int flags, void*)
{
	switch (event) {
	case EVENT_LBUTTONDOWN:
		g_pointOld = Point(x, y);
		cout << "EVENT_LBUTTONDOWN: " << x << ", " << y << endl;
		break;
	case EVENT_LBUTTONUP:
		cout << "EVENT_LBUTTONUP: " << x << ", " << y << endl;
		break;
	case EVENT_MOUSEMOVE:
		if (flags & EVENT_FLAG_LBUTTON) {
			line(g_img, g_pointOld, Point(x, y), Scalar(0, 255, 255), 2);
			imshow("img", g_img);
			g_pointOld = Point(x, y);
		}
		break;
	default:
		break;
	}
}

void Trackbar()
{
	Mat img = Mat::zeros(400, 400, CV_8UC1);

	namedWindow("image");
	createTrackbar("level", "image", 0, 16, OnLevelChange, (void*)&img);

	imshow("image", img);
	waitKey();
}

void OnLevelChange(int pos, void* userdata)
{
	Mat img = *(Mat*)userdata;

	img.setTo(pos * 16);
	imshow("image", img);
}

void MaskSetTo()
{
	Mat src = imread("lenna.bmp", IMREAD_COLOR);
	Mat mask = imread("mask_smile.bmp", IMREAD_GRAYSCALE);

	if (src.empty() || mask.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	src.setTo(Scalar(0, 255, 255), mask);

	imshow("src", src);
	imshow("mask", mask);

	waitKey(0);
	destroyAllWindows();
}

void MaskCopyTo()
{
	Mat src = imread("airplane.bmp", IMREAD_COLOR);
	Mat mask = imread("mask_plane.bmp", IMREAD_GRAYSCALE);
	Mat dst = imread("field.bmp", IMREAD_COLOR);

	if (src.empty() || mask.empty() || dst.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	src.copyTo(dst, mask);

	imshow("src", src);
	imshow("dst", dst);
	imshow("mask", mask);

	waitKey(0);
	destroyAllWindows();
}

void TimeInverse()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dst(src.rows, src.cols, src.type());

	TickMeter tm;
	tm.start();

	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			dst.at<uchar>(j, i) = 255 - src.at<uchar>(j, i);
		}
	}

	tm.stop();
	cout << "Image inverse took " << tm.getTimeMilli() << "ms." << endl;
}

void UsefulFunc()
{
	Mat img = imread("lenna.bmp", IMREAD_GRAYSCALE);

	cout << "Sum: " << (int)sum(img)[0] << endl;
	cout << "Mean: " << (int)mean(img)[0] << endl;

	double minVal, maxVal;
	Point minPos, maxPos;
	minMaxLoc(img, &minVal, &maxVal, &minPos, &maxPos);

	cout << "minVal: " << minVal << " at " << minPos << endl;
	cout << "maxVal: " << maxVal << " at " << maxPos << endl;

	Mat src = Mat_<float>({ 1, 5 }, { -1.f, -0.5f, 0.f, 0.5f, 1.f });

	Mat dst;
	normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);

	cout << "src: " << src << endl;
	cout << "dst: " << dst << endl;

	cout << "cvRound(2.5): " << cvRound(2.5) << endl;
	cout << "cvRound(2.51): " << cvRound(2.51) << endl;
	cout << "cvRound(3.4999): " << cvRound(3.4999) << endl;
	cout << "cvRound(3.5): " << cvRound(3.5) << endl;
}

void Brightness1()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dst = src + 100;

	imshow("src", src);
	imshow("dst", dst);
	waitKey();

	destroyAllWindows();
}

void BrightnessTrackbar()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "image load failed!" << endl;
		return;
	}

	namedWindow("dst");
	createTrackbar("Brightness", "dst", 0, 100, OnBrightness, (void*)&src);

	waitKey();
	destroyAllWindows();
}

void OnBrightness(int pos, void* userdata)
{
	Mat src = *(Mat*)userdata;
	Mat dst = src + pos;

	imshow("dst", dst);
}

void Contrast()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	float alpha = 1.f;
	Mat dst = src + (src - 128) * alpha;

	imshow("src", src);
	imshow("dst", dst);
	waitKey();

	destroyAllWindows();
}

Mat CalcGrayHist(const Mat& img)
{
	CV_Assert(img.type() == CV_8UC1);

	Mat hist;
	int channels[] = { 0 };
	int dims = 1;
	const int histSize[] = { 256 };
	float graylevel[] = { 0, 256 };
	const float* ranges[] = { graylevel };

	calcHist(&img, 1, channels, noArray(), hist, dims, histSize, ranges);

	return hist;
}

Mat GetGrayHistImage(const Mat& hist)
{
	CV_Assert(hist.type() == CV_32FC1);
	CV_Assert(hist.size() == Size(1, 256));

	double histMax;
	minMaxLoc(hist, 0, &histMax);

	Mat imgHist(100, 256, CV_8UC1, Scalar(255));
	for (int i = 0; i < 256; i++) {
		line(imgHist, Point(i, 100),
			Point(i, 100 - cvRound(hist.at<float>(i, 0) * 100 / histMax)), Scalar(0));
	}

	return imgHist;
}

void HistogramStreching()
{
	Mat src = imread("hawkes.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	double gmin, gmax;
	minMaxLoc(src, &gmin, &gmax);

	Mat dst = (src - gmin) * 255 / (gmax - gmin);

	imshow("src", src);
	imshow("srcHist", GetGrayHistImage(CalcGrayHist(src)));

	imshow("dst", dst);
	imshow("dstHist", GetGrayHistImage(CalcGrayHist(dst)));

	waitKey();
	destroyAllWindows();
}

void HistogramEqualization()
{
	Mat src = imread("hawkes.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dst;
	equalizeHist(src, dst);

	imshow("src", src);
	imshow("srcHist", GetGrayHistImage(CalcGrayHist(src)));

	imshow("dst", dst);
	imshow("dstHist", GetGrayHistImage(CalcGrayHist(dst)));

	waitKey();
	destroyAllWindows();
}

void Arithmetic()
{
	Mat src1 = imread("lenna256.bmp", IMREAD_GRAYSCALE);
	Mat src2 = imread("square.bmp", IMREAD_GRAYSCALE);

	if (src1.empty() || src2.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	imshow("src1", src1);
	imshow("src2", src2);

	Mat dst1, dst2, dst3, dst4;

	add(src1, src2, dst1);
	addWeighted(src1, 0.5, src2, 0.5, 0, dst2);
	subtract(src1, src2, dst3);
	absdiff(src1, src2, dst4);

	imshow("dst1", dst1);
	imshow("dst2", dst2);
	imshow("dst3", dst3);
	imshow("dst4", dst4);
	waitKey();
}

void Logical()
{
	Mat src1 = imread("lenna256.bmp", IMREAD_GRAYSCALE);
	Mat src2 = imread("square.bmp", IMREAD_GRAYSCALE);

	if (src1.empty() || src2.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	imshow("src1", src1);
	imshow("src2", src2);

	Mat dst1, dst2, dst3, dst4;

	bitwise_and(src1, src2, dst1);
	bitwise_or(src1, src2, dst2);
	bitwise_xor(src1, src2, dst3);
	bitwise_not(src1, dst4);

	imshow("dst1", dst1);
	imshow("dst2", dst2);
	imshow("dst3", dst3);
	imshow("dst4", dst4);
	waitKey();
}