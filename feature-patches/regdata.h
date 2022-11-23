#ifndef REG_DATA_H
#define REG_DATA_H

#include <string>
#include <vector>
#include <Eigen/Dense>

struct RegInputData {                       // assumes no lens distortion
    int width, height;                      // size of input frame
    Eigen::Matrix3d deviceIntrinsicMatrix;  // input camera intrinsics
    Eigen::Matrix4d deviceExtrinsicMatrix;  // ARKit/ARCore extrinsics matrix
};

struct RegResultData {
    std::string camera_model;         // "SIMPLE_RADIAL" or "PINHOLE"
    std::vector<double> distCoeffs;   // [K1] for "SIMPLE_RADIAL"
    Eigen::Matrix4d colmap2world;     // colmap -> PGAT space
    Eigen::Matrix3d intrinsicMatrix;  // (solved) camera intrinsic matrix
    Eigen::Matrix4d extrinsicMatrix;  // (solved) camera extrinsic matrix
    Eigen::Matrix4d correctionMatrix; // corection of deviceExtrinsicMatrix
};

//
// Read foo.json which contains frame metadata sent to regserver.
//
void readRegInputJSON(std::string jsonPath, RegInputData& regData);

//
// Read foo_corr.json returned from successful registration.
//
void readRegResultJSON(std::string jsonPath, RegResultData& regData);

#endif // REG_DATA_H
