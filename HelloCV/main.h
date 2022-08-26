#pragma once
#include "opencv2/opencv.hpp"
#include <vector>

using namespace cv;

//int GetNumber();
//void Run(int iNumber);

void MatOp();
void Camera_in();
void VideoIn();

void DrawLines();
void DrawText();

void Keyboard();
void Mouse();
void OnMouse(int event, int x, int y, int flags, void*);
void Trackbar();
void OnLevelChange(int pos, void* userdata);

void MaskSetTo();
void MaskCopyTo();
void TimeInverse();
void UsefulFunc();

// Brightness & Contrast
void Brightness1();
void BrightnessTrackbar();
void OnBrightness(int pos, void* userdata);
void Contrast();
Mat CalcGrayHist(const Mat& img);
Mat GetGrayHistImage(const Mat& hist);
void HistogramStreching();
void HistogramEqualization();

// Arithmetic & Logical
void Arithmetic();
void Logical();

// Filtering
void FilterEmbossing();
void BlurringMean();
void BlurringGaussian();
void UnsharpMask();
void NoiseGaussian();
void FilterBilateral();
void FilterMedian();

// Geometric Transform
void AffineTransform();
void AffineTranslation();
void AffineShear();
void AffineScale();
void AffineRotation();
void AffineFlip();
void OnMouse2(int event, int x, int y, int flags, void* userdata);
void Perspective();

// Edge & Line & Circle Detect
void SobelEdge();
void CannyEdge();
void HoughLines();
void HoughLineSegments();
void HoughCircles();

// Color
void ColorInverse();
void ColorGrayscale();
void ColorSplit();
void ColorEqHist();
void InRange();
void OnHueChanged(int, void*);
void BackProject();

// Binarize & Morphology
void Binarize(int argc, char* argv[]);
void OnThreshold(int, void*);
void Adaptive();
void OnTrackbar(int pos, void* userdata);
void ErodeDilate();
void OpenClose();

// Labeling & Contour Detect
void LabelingBasic();
void LabelingStats();
void ContoursBasic();
void ContoursHier();
void SetLabel(Mat&, const std::vector<Point>&, const String&);
void Polygon();

// Object Detect
void TemplateMatching();
void DetectFace();
void DetectEyes();
void Hog();
void DecodeQRCode();

// Feature point Detect and Matching (keypoint, interest point)
void CornerHarris();
void CornerFAST();
void DetectKeypoints();
void KeypointMatching();
void GoodMatching();
void FindHomography();
void Stitching(int argc, char* argv[]);