#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include "regdata.h"

using nlohmann::json;

void readRegInputJSON(std::string jsonPath, RegInputData& regData) {
    std::ifstream is(jsonPath);
    if (!is.is_open()) {
        std::cerr << "Unable to open '" << jsonPath << "'\n";
        exit(-1);
    }
    json image_json;
    try {
        is >> image_json;
    } catch (json::parse_error) {
        std::cerr << "Fail to parse '"  << jsonPath << "'\n";
        exit(-1);
    }

    regData.width = regData.height = 0;
    if (image_json.find("img_width") != image_json.end()) {
        regData.width = image_json["img_width"].get<int>();
    } else if (image_json.find("imgWidth") != image_json.end()) {
        regData.width = image_json["imgWidth"].get<int>();
    } else {
        std::cerr << "No 'image_width' key in '" << jsonPath << "'\n";;
        exit(-1);
    }
        
    if (image_json.find("img_height") != image_json.end()) {
        regData.height = image_json["img_height"].get<int>();
    } else if (image_json.find("imgHeight") != image_json.end()) {
        regData.height = image_json["imgHeight"].get<int>();
    } else {
        std::cerr << "No 'image_height' key in '" << jsonPath << "'\n";;
        exit(-1);
    }
    
    std::vector<double> mat4x4;
    if (image_json.find("cam_extrinsics") != image_json.end()) {
        mat4x4 = image_json["cam_extrinsics"].get<std::vector<double>>();
    } else if (image_json.find("camXform") != image_json.end()) {
        mat4x4 = image_json["camXform"].get<std::vector<double>>();
    } else {
        std::cerr << "No 'camEntrinsics' key in '" << jsonPath << "'\n";;
        exit(-1);
    }
    assert(mat4x4.size() == 16);
    regData.deviceExtrinsicMatrix = Eigen::Matrix4d(mat4x4.data()); // col-major order

    std::vector<double> mat3x3;
    if (image_json.find("camIntrinsics") != image_json.end()) {
        mat3x3 = image_json["camIntrinsics"].get<std::vector<double>>();
    } else if (image_json.find("cam_intrinsics") != image_json.end()) {
        mat3x3 = image_json["cam_intrinsics"].get<std::vector<double>>();
    } else {
        std::cerr << "No 'camIntrinsics' key in '" << jsonPath << "'\n";;
        exit(-1);
    }
    assert(mat3x3.size() == 9);
    regData.deviceIntrinsicMatrix = Eigen::Matrix3d(mat3x3.data());
}


void readRegResultJSON(std::string jsonPath, RegResultData& regData) {
    std::ifstream is(jsonPath);
    if (!is.is_open()) {
        std::cerr << "Unable to open '" << jsonPath << "'\n";
        exit(-1);
    }
    json corr_json;
    try {
        is >> corr_json;
    } catch (json::parse_error) {
        std::cerr << "Fail to parse '"  << jsonPath << "'\n";
        exit(-1);
    }

    std::vector<double> mat4x4;
    if (corr_json.find("model_to_world") != corr_json.end()) {
        mat4x4 = corr_json["model_to_world"].get<std::vector<double>>();
    } else  if (corr_json.find("colmap2world") != corr_json.end()) {
        mat4x4 = corr_json["colmap2world"].get<std::vector<double>>();
    } else {
        std::cerr << "No 'colmap2world' key in '" << jsonPath << "'\n";;
        exit(-1);
    }
    assert(mat4x4.size() == 16);
    regData.colmap2world = Eigen::Matrix4d(mat4x4.data()); // col-major order

    if (corr_json.find("camera_model") == corr_json.end()) {
        std::cerr << "No 'camera_model' key in '" << jsonPath << "'\n";;
        exit(-1);
    }
    regData.camera_model = corr_json["camera_model"].get<std::string>();

    if (corr_json.find("camera_params") == corr_json.end()) {
        std::cerr << "No 'camera_params' key in '" << jsonPath << "'\n";;
        exit(-1);
    }
    std::vector<double> camera_params
        = corr_json["camera_params"].get<std::vector<double>>();
    assert(camera_params.size() == 4);
    if (regData.camera_model == "PINHOLE") {
        const double FX = camera_params[0];
        const double FY = camera_params[1];
        const double CX = camera_params[2];
        const double CY = camera_params[3];
        regData.intrinsicMatrix
            << FX, 0, CX,
            0, FY, CY,
            0, 0,  1;
    } else if (regData.camera_model == "SIMPLE_RADIAL") {
        const double F = camera_params[0];
        const double CX = camera_params[1];
        const double CY = camera_params[2];
        const double K1 = camera_params[3];
        regData.intrinsicMatrix
            << F, 0, CX,
            0, F, CY,
            0, 0,  1;
        regData.distCoeffs.push_back(K1);
    } else {
        std::cerr << "Unsupported 'camera_model' in '" << jsonPath << "'\n";;
        exit(-1);
    }

    std::vector<double> pose;
    if (corr_json.find("model_pose") != corr_json.end()) {
        pose = corr_json["model_pose"].get<std::vector<double>>();
    } else if (corr_json.find("colmap_cam") != corr_json.end()) {
        pose = corr_json["colmap_cam"].get<std::vector<double>>();
    } else {
        std::cerr << "No 'colmap_cam' key in '" << jsonPath << "'\n";;
        exit(-1);
    }
    assert(pose.size() == 7);

    Eigen::Quaterniond Q(pose[0], pose[1], pose[2], pose[3]);
    Eigen::Matrix3d R = Q.normalized().toRotationMatrix();
    Eigen::Vector3d T(pose[4], pose[5], pose[6]);
    regData.extrinsicMatrix = Eigen::Matrix4d::Identity();
    regData.extrinsicMatrix.block<3,3>(0,0) = R;
    regData.extrinsicMatrix.block<3,1>(0,3) = T;
    
    if (corr_json.find("correction") == corr_json.end()) {
        std::cerr << "No 'correction' key in '" << jsonPath << "'\n";;
        exit(-1);
    }
    mat4x4 = corr_json["correction"].get<std::vector<double>>();
    assert(mat4x4.size() == 16);
    regData.correctionMatrix = Eigen::Matrix4d(mat4x4.data()); // col-major order
}

