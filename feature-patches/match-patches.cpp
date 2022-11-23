#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>
#include "regdata.h"

using nlohmann::json;
using namespace cv;

//
// Simple global patch matching.
// Given:
//   * target image
//   * image camera pose + intrinsics
//   * image feature point data + assoc. 3D points
//   * set of patches + metadata
// Output:
//   Finds the best matching patches in the target image in the region
//   where the 3D points projects onto the image,
//   and if the match is sufficient we output
//   * list of matching patch locations
//   * draw rectangles on matches in target image
//
//   * include approx. image pose + camera instrincs to
//     predict patch locations via projection of 3D points.
//

int main(int argc, char *argv[]) {

    if (argc != 8) {
        std::cerr << "usage: " << argv[0]
                  << " image.jgp image.json image_corr.json patches.jpg patches.json matches.jpg matches.json\n";
        exit(1);
    }

    const std::string targetImagePath(argv[1]);
    const std::string targetImageJSONPath(argv[2]);
    const std::string regJSONPath(argv[3]);
    const std::string patchesImagePath(argv[4]);
    const std::string patchesJsonPath(argv[5]);
    const std::string matchesImagePatch(argv[6]);
    const std::string matchesJsonPath(argv[7]);

    Mat image = imread(targetImagePath, IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Unable to open input image '"
                  << targetImagePath << "'!\n";
        exit(-1);
    }
    Rect imageRect(0, 0, image.cols, image.rows);

    Mat grayImage;
    cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    RegInputData imageData;
    readRegInputJSON(targetImageJSONPath, imageData);

    RegResultData regData;
    readRegResultJSON(regJSONPath, regData);

    Eigen::Matrix4d rotateYDown{{1,0,0,0}, {0,-1,0,0}, {0,0,-1,0}, {0,0,0,1}};
    Eigen::Matrix4d correctedExtrinsicTransform =
        rotateYDown * imageData.deviceExtrinsicMatrix.inverse() * regData.correctionMatrix;
    
    Mat patchesImage = imread(patchesImagePath, IMREAD_COLOR);
    if (patchesImage.empty()) {
        std::cerr << "Unable to open input patches image '"
                  << patchesImagePath << "'!\n";
        exit(-1);
    }

    Mat patchesGrayImage;
    cvtColor(patchesImage, patchesGrayImage, cv::COLOR_BGR2GRAY);

    struct PatchInfo {
        Rect packedRect;
        Point2d keypoint;
        Point2d keypointOffset;
        Point3d point3d;
        Point2d point2d;
    };

    std::vector<PatchInfo> patchData;
    
    {
        std::ifstream is(patchesJsonPath);
        if (!is.is_open()) {
            std::cerr << "Unable to open '" << patchesJsonPath << "'\n";
            exit(-1);
        }
        json patches_json;
        try {
            is >> patches_json;
        } catch (json::parse_error) {
            std::cerr << "Fail to parse '"  << patchesJsonPath << "'\n";
            exit(-1);
        }
        is.close();

        assert(patches_json.is_array());
        for (auto&& pj : patches_json) {
            assert(pj.find("keypoint") != pj.end());
            assert(pj["keypoint"].is_array());
            std::vector<double> keypoint = pj["keypoint"].get<std::vector<double>>();
            
            assert(pj.find("patch_pos") != pj.end());
            assert(pj["patch_pos"].is_array());
            std::vector<int> patchPos = pj["patch_pos"].get<std::vector<int>>();

            const double kOffsetX = keypoint[0] - patchPos[0];
            const double kOffsetY = keypoint[1] - patchPos[1];

            assert(pj.find("packed_pos") != pj.end());
            assert(pj["packed_pos"].is_array());
            std::vector<int> packedPos = pj["packed_pos"].get<std::vector<int>>();
            assert(pj.find("patch_size") != pj.end());
            assert(pj["patch_size"].is_array());
            std::vector<int> psize = pj["patch_size"].get<std::vector<int>>();
            Rect rect(packedPos[0], packedPos[1], psize[0], psize[1]);

            assert(pj.find("patch_point3D") != pj.end());
            assert(pj["patch_point3D"].is_array());
            std::vector<double> pp = pj["patch_point3D"].get<std::vector<double>>();
            auto P = correctedExtrinsicTransform * Eigen::Vector4d(pp[0], pp[1], pp[2], 1);

            PatchInfo patchInfo;
            patchInfo.packedRect = rect;
            patchInfo.keypoint = Point2d(keypoint[0], keypoint[1]);
            patchInfo.keypointOffset = Point2d(kOffsetX, kOffsetY);
            patchInfo.point3d = Point3d(P.x(),P.y(),P.z());
            patchData.emplace_back(std::move(patchInfo));
        }
    }

    // clip points3D to view frustum ??
    
    Mat rot = Mat::eye(3,3,CV_64F);
    Mat rotVec = Mat::zeros(3,1,CV_64F);
    Rodrigues(rotVec, rot);
    auto transVec = Vec<double,3>::zeros();
    Mat K = Mat::eye(3,3,CV_64F);
    K.at<double>(0,0) = imageData.deviceIntrinsicMatrix(0,0);
    K.at<double>(1,1) = imageData.deviceIntrinsicMatrix(1,1);
    K.at<double>(0,2) = imageData.deviceIntrinsicMatrix(0,2);
    K.at<double>(1,2) = imageData.deviceIntrinsicMatrix(1,2);
    static const std::vector<double> distCoeffs = {0,0,0,0};
    
    std::vector<Point3d> points3D(patchData.size());
    std::transform(patchData.begin(), patchData.end(), points3D.begin(),
                   [](const PatchInfo& p) -> Point3d {
                       return p.point3d;
                   });

    std::vector<Point2d> imagePoints(points3D.size());
    projectPoints(points3D, rotVec, transVec, K, distCoeffs, imagePoints);

    for (size_t i = 0; i < imagePoints.size(); i++)
        patchData[i].point2d = imagePoints[i];
    
    struct MatchInfo {
        double matchVal;
        Point matchLoc;
        bool success;
        MatchInfo(double val, Point loc, bool s)
            : matchVal{val}, matchLoc{loc}, success{s} {}
    };
    std::vector<MatchInfo> matchesInfo;

    constexpr int searchRadius = 80; // amount to expand box for search region

    double matchesValuesSum = 0;
    double matchesValuesSquaredSum = 0;
    int matchesSum = 0;

    for (auto&& patchInfo : patchData) {
        const auto packedRect = patchInfo.packedRect;
        Mat patch = patchesGrayImage(packedRect);

        Rect targetRect(int(std::floor(patchInfo.point2d.x - packedRect.width/2.0)),
                        int(std::floor(patchInfo.point2d.y - packedRect.height/2.0)),
                        packedRect.width, packedRect.height);
        
        Rect searchRect = Rect(targetRect.x - searchRadius,
                               targetRect.y - searchRadius,
                               targetRect.width + 2*searchRadius,
                               targetRect.height + 2*searchRadius);
        Rect clampedSearchRect = imageRect & searchRect;

        if (clampedSearchRect.width >= targetRect.width &&
            clampedSearchRect.height >= targetRect.height) {
            Mat result;
            matchTemplate(grayImage(clampedSearchRect), patch, result, TM_CCOEFF_NORMED);
            double minVal, maxVal;
            Point minLoc, maxLoc;
            minMaxLoc(result, &minVal, &maxVal, &minLoc,  &maxLoc);
            matchesValuesSum += maxVal;
            matchesValuesSquaredSum += maxVal*maxVal;
            matchesSum++;
            matchesInfo.emplace_back(MatchInfo{maxVal, maxLoc, true});
            Rect matchRect{clampedSearchRect.x + maxLoc.x,
                           clampedSearchRect.y + maxLoc.y,
                           packedRect.width, packedRect.height};
        
            constexpr int thickness = 2;
            const Scalar targetColor(255, 255, 0); // BGR
            const Scalar matchColor(0, 255, 255);  // BGR
            rectangle(image, targetRect, targetColor, thickness);
            rectangle(image, matchRect, matchColor, thickness);
        } else {
            matchesInfo.emplace_back(MatchInfo{0, Point(0,0), false});
        }
    }

    const double matchesValuesMean = matchesValuesSum / matchesSum;
    const double matchesValuesVar = matchesValuesSquaredSum/matchesSum - matchesValuesMean*matchesValuesMean;
    const double matchesValuesStdDev = std::sqrt(matchesValuesVar);
    std::cout << "Matches value mean  = " << matchesValuesMean << "\n";
    std::cout << "Matches value stdev = " << matchesValuesStdDev << "\n";

    imwrite(matchesImagePatch, image);

    json matchesData = json::array();
    for (auto&& matches : matchesInfo) {
        json j = json::object();
        j["match_val"] = matches.matchVal;
        j["match_location"] = {matches.matchLoc.x, matches.matchLoc.y};
        j["match_success"] = matches.success;
        matchesData.push_back(j);
    }

    std::ofstream js(matchesJsonPath);
    if (!js.is_open()) {
        std::cerr << "Unable to open '" << matchesJsonPath << "' for writing!\n";
        exit(-1);
    }
    js << matchesData;
    js.close();

    return 0;
}
