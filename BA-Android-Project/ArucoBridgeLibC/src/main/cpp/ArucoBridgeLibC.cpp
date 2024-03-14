#include <jni.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <vector>
#include <string>
#include <opencv2/calib3d.hpp>


using namespace cv;

extern "C" {
    //Good Features
    Mat cameraFrameFeatures;

    //Aruco
    Mat cameraFrameAruco;
    Mat markerImage;
    Mat markerImageConverted;

    aruco::DetectorParameters detectorParams = aruco::DetectorParameters();
    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    aruco::ArucoDetector detector(dictionary, detectorParams);

    //Camera Calibrations
    int cameraWidth, cameraHeight, max_d;
    float cx, cy, fx, fy;
    Mat kMat, distCoeffs;
    double _dc[] = {0,0,0,0}; //no distortion
    Mat modelMarkerCorners;

    void InitOpenFrame(int width, int height){
        cameraWidth = width;
        cameraHeight = height;

        //Pose Estimation Setup
        max_d = max(cameraHeight, cameraWidth);
        cx = cameraWidth / 2.0f;
        cy = cameraHeight / 2.0f;
        fx= max_d;
        fy = max_d;

        kMat = (Mat_<double>(3,3) <<
                fx, 0., cx,
                0., fy, cy,
                0., 0., 1.);

        //distortion Coefficients
        distCoeffs = Mat(1, 4, CV_64FC1, _dc);

        float kMarkerSize = 0.025f; //= 2.5cm
        float pos0 = kMarkerSize / 2.0f;
        modelMarkerCorners = Mat(4, 1, CV_32FC3);
        modelMarkerCorners.ptr<Vec3f>(0)[0] = Vec3f(-pos0, pos0, 0);
        modelMarkerCorners.ptr<Vec3f>(0)[1] = Vec3f(pos0, pos0, 0);
        modelMarkerCorners.ptr<Vec3f>(0)[2] = Vec3f(pos0, -pos0, 0);
        modelMarkerCorners.ptr<Vec3f>(0)[3] = Vec3f(-pos0, -pos0, 0);


        cameraFrameFeatures = Mat(height, width, CV_8UC4);
        cameraFrameAruco = Mat(height, width, CV_8UC4);

        aruco::generateImageMarker(dictionary, 23, 200, markerImage, 1);
        cvtColor(markerImage, markerImageConverted, COLOR_GRAY2RGBA);

    }

    void GetFeatures(unsigned char** rawImage){
        cameraFrameFeatures.data = *rawImage;

        Mat grey;
        cvtColor(cameraFrameFeatures, grey, COLOR_RGBA2GRAY);

        std::vector<Point2f> corners;
        goodFeaturesToTrack(grey, corners, 20, 0.01,
                            10, Mat(), 3,
                            false, 0.04);

        for(int i = 0; i < corners.size(); ++i){
            circle(cameraFrameFeatures, corners[i], 8,
                   Scalar(0, 255, 0, 0), 1);
        }
    }

    int GetArucoDrawing(unsigned char** rawImage){
        cameraFrameAruco.data = *rawImage;

        Mat tempMat;
        cvtColor(cameraFrameAruco, tempMat, COLOR_RGBA2RGB);


        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
        detector.detectMarkers(tempMat, markerCorners,
                               markerIds, rejectedCandidates);


        //Draw Corners of rejected Candidates red
        for(int i = 0; i < rejectedCandidates.size(); i++){
            for(int j = 0; j < rejectedCandidates[i].size(); j++){
                circle(cameraFrameAruco, rejectedCandidates[i][j], 8,
                      Scalar(255, 0, 0, 0), 1);
            }
        }

        //Draw Corners of detected markers green
        for(int i = 0; i < markerCorners.size(); i++){
            for(int j = 0; j < markerCorners[i].size(); j++){
                circle(cameraFrameAruco, markerCorners[i][j], 8,
                       Scalar(0, 255, 0, 0), 1);
            }
        }

        if(markerIds.empty()) return -1;

        //Pose Estimation
        int nMarkers = markerCorners.size();
        std::vector<Vec3d> rvecs(nMarkers), tvecs(nMarkers);

        for(int i = 0; i < markerIds.size(); i++){
            solvePnP(modelMarkerCorners,
                     markerCorners.at(i),
                     kMat, distCoeffs,
                     rvecs.at(i), tvecs.at(i),
                     false);
        }

        for(int i = 0; i < rvecs.size(); i++){
            auto rvec = rvecs[i];
            auto tvec = tvecs[i];

            drawFrameAxes(cameraFrameAruco,
                          kMat, distCoeffs,
                          rvec, tvec,
                          0.03);
        }


        return 1;

    }
}