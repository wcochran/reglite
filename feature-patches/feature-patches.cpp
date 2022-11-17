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
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include "rectpack2D/finders_interface.h"

using nlohmann::json;
using namespace cv;
using namespace rectpack2D;

struct PatchInfo {
    Eigen::Vector2f keypoint;         // position of feature in original image
    cv::Rect rect;                    // location and size of patch in original image
    rectpack2D::rect_xywh packedRect; // location and size of patch in packed rectangle image
    Eigen::Vector3f point3D;          // corresponding 3D location (unused at the moment)
    PatchInfo(float x, float y,
              cv::Rect rect,
              rectpack2D::rect_xywh patch,
              float X = 0, float Y = 0, float Z = 0)
    : keypoint{x,y}, rect{rect}, packedRect{patch}, point3D{X,Y,Z} {}
};

void to_json(json& j, const PatchInfo& patch) {
    j["keypoint"] = {patch.keypoint.x(), patch.keypoint.y()};
    j["patch_pos"] = {patch.rect.x, patch.rect.y};
    j["patch_size"] = {patch.rect.width, patch.rect.height};
    j["packed_pos"] = {patch.packedRect.x, patch.packedRect.y};
    j["patch_point3D"] = {patch.point3D.x(), patch.point3D.y(), patch.point3D.z()};
}

int main(int argc, char *argv[]) {

    if (argc != 5) {
        std::cerr << "usage: " << argv[0]
                  << " image_corr.json image.jpg patches.jpg patches.json\n";
        exit(1);
    }

    std::string regdata_json_path(argv[1]);
    std::string image_path(argv[2]);
    std::string patches_image_path(argv[3]);
    std::string patches_metadata_path(argv[4]);

    Mat image = imread(image_path, IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Unable to open input image '"
                  << image_path << "'!\n";
        exit(-1);
    }
    Rect imageRect(0, 0, image.cols, image.rows);
    
    std::vector<Point2f> closestPoints;
    std::vector<std::array<float,4>> closestShapes;
    
    {
        std::ifstream is(regdata_json_path);
        if (!is.is_open()) {
            std::cerr << "Unable to open '" << regdata_json_path << "'\n";
            exit(-1);
        }
        json camera_json;
        try {
            is >> camera_json;
        } catch (json::parse_error) {
            std::cerr << "Fail to parse '"  << regdata_json_path << "'\n";
            exit(-1);
        }

        if (camera_json.find("closestPoints") == camera_json.end() ||
            camera_json.find("closestFeatureShapes") == camera_json.end()) {
            std::cerr << "Keys 'closestPoints' or 'closestFeatureShapes' not found!\n";
            exit(2);
        }
        
        std::vector<float> flattenedPoints =
            camera_json["closestPoints"].get<std::vector<float>>();
        for (size_t i = 0; i < flattenedPoints.size(); i += 2) {
            const float x = flattenedPoints[i];
            const float y = flattenedPoints[i+1];
            closestPoints.emplace_back(Point2f(x,y));
        }
        
        std::vector<float> flattenedShapes = camera_json["closestFeatureShapes"].get<std::vector<float>>();
        for (size_t i = 0; i < flattenedShapes.size(); i += 4) {
            std::array<float,4> a;
            std::copy(&flattenedShapes[i], &flattenedShapes[i+4], a.begin());
            closestShapes.emplace_back(a);
        }
    }

    assert(closestPoints.size() == closestShapes.size());

    std::vector<Mat> patches;
    std::vector<PatchInfo> patchMetadata;

	constexpr bool allow_flip = false;
	const auto runtime_flipping_mode = flipping_option::DISABLED;
	using spaces_type = rectpack2D::empty_spaces<allow_flip, default_empty_spaces>;
	using rect_type = output_rect_t<spaces_type>;
	std::vector<rect_type> rectangles;

    constexpr int padding = 4;
    constexpr double bboxScale = 1.414;

    for (size_t i = 0; i < closestPoints.size(); i++) {
        const Point2f& p = closestPoints[i];
        const std::array<float,4>& shape = closestShapes[i];
        const Eigen::Matrix2f A{{shape[0], shape[1]},
                                {shape[2], shape[3]}};
        const double scaleX = A.col(0).norm() * bboxScale;
        const double scaleY = A.col(1).norm() * bboxScale;
        const double orientation = std::atan2(A(1,0), A(0,0));
        const double c = cos(orientation), s = sin(orientation);
        const Eigen::Vector2d u = scaleX * Eigen::Vector2d(c, s);
        const Eigen::Vector2d v = scaleY * Eigen::Vector2d(-s, c);
        // https://iquilezles.org/articles/ellipses/
        const double bboxWidth  = std::sqrt(u.x()*u.x() + v.x()*v.x());
        const double bboxHeight = std::sqrt(u.y()*u.y() + v.y()*v.y());
        const double bboxLeft  = p.x - bboxWidth;
        const double bboxRight = p.x + bboxWidth;
        const double bboxTop = p.y - bboxHeight;
        const double bboxBottom = p.y + bboxHeight;
        // Note: I think x0,y0 can be non-integral (?)
        const int x0 = int(std::floor(bboxLeft));
        const int y0 = int(std::floor(bboxTop));
        const int W = int(std::ceil(bboxRight)) - x0;
        const int H = int(std::ceil(bboxBottom)) - y0;
        if (W < 2 || H < 2) continue;
        Rect rect(x0, y0, W, H);
        const bool is_inside = (rect & imageRect) == rect;
        if (!is_inside) continue;
        Mat patch = image(rect);
        patches.emplace_back(std::move(patch));
        rect_xywh packedRect(0,0,W+padding,H+padding);
        rectangles.emplace_back(packedRect);
        PatchInfo info{p.x, p.y, rect, packedRect};
        patchMetadata.emplace_back(info);
    }

	const auto max_side = 1000;
	const auto discard_step = -4;
    bool packing_success = true;
    const auto result_size = find_best_packing<spaces_type>(
        rectangles,
        make_finder_input(
            max_side,
            discard_step,
            [](rect_type& r) {
                return callback_result::CONTINUE_PACKING;
            },
            [&](rect_type& r) {
                packing_success = false;
                return callback_result::ABORT_PACKING;
            },
            runtime_flipping_mode
			)
		);

    if (!packing_success) {
        std::cerr << "Unable to pack rectangles!\n";
        exit(-1);
    }

    std::multimap<std::pair<int,int>,size_t> rectToIndex;
    for (size_t i = 0; i < rectangles.size(); i++) {
        const auto& rect = rectangles[i];
        rectToIndex.insert({std::make_pair(rect.w,rect.h),i});
    }

    assert(patches.size() == patchMetadata.size());

    Mat packedPatches(result_size.h, result_size.w, CV_8UC3, Scalar(0, 0, 0));
    for  (size_t i = 0; i < patches.size(); i++) {
        auto& patch = patches[i];
        auto& patchInfo = patchMetadata[i];
        const auto wh = std::make_pair(patch.cols + padding, patch.rows + padding);
        auto iter = rectToIndex.find(wh);
        assert(iter != rectToIndex.end());
        const size_t index = iter->second;
        const auto rect = rectangles[index];
//        std::cout << "rect: " << rect.x << "," << rect.y << ":" << rect.w << "x" << rect.h << std::endl;
        Rect roi(rect.x,rect.y,rect.w - padding,rect.h - padding);
        patch.copyTo(packedPatches(roi));
        rectToIndex.erase(iter);
        patchInfo.packedRect = rect;
    }

    imwrite(patches_image_path, packedPatches);

    json patchData = json::array();
    for (auto&& patchInfo : patchMetadata) {
        json j = json::object();
        to_json(j, patchInfo);
        patchData.push_back(j);
    }

    std::ofstream js(patches_metadata_path);
    if (!js.is_open()) {
        std::cerr << "Unable to open '" << patches_metadata_path << "' for writing!\n";
        exit(-1);
    }
    js << patchData;
    js.close();

    return 0;
}
