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

int main(int argc, char *argv[]) {

    if (argc != 4) {
        std::cerr << "usage: " << argv[0]
                  << " image_corr.json image.jpg patches.jpg\n";
        exit(1);
    }

    std::string regdata_json_path(argv[1]);
    std::string image_path(argv[2]);
    std::string patches_path(argv[3]);

    Mat image = imread(image_path, IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Unable to open input image '"
                  << image_path << "'!\n";
        exit(-1);
    }
    
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

	constexpr bool allow_flip = false;
	const auto runtime_flipping_mode = flipping_option::DISABLED;
	using spaces_type = rectpack2D::empty_spaces<allow_flip, default_empty_spaces>;
	using rect_type = output_rect_t<spaces_type>;
	std::vector<rect_type> rectangles;

    for (size_t i = 0; i < closestPoints.size(); i++) {
        const Point2f& p = closestPoints[i];
        const std::array<float,4>& shape = closestShapes[i];
        const Eigen::Matrix2f A{{shape[0], shape[1]},
                                {shape[2], shape[3]}};
        const double scaleX = A.col(0).norm();
        const double scaleY = A.col(1).norm();
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
        const int x0 = int(std::round(bboxLeft));
        const int y0 = int(std::round(bboxTop));
        const int W = int(std::round(bboxRight)) - x0;
        const int H = int(std::round(bboxBottom)) - y0;
        Rect rect(x0, y0, W, H);
        Mat patch = image(rect);
        patches.emplace_back(std::move(patch));
        rectangles.emplace_back(rect_xywh(0,0,W,H));
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

    Mat packedPatches(result_size.h, result_size.w, CV_8UC3, Scalar(0, 0, 0));
    for (auto&& patch : patches) {
        const auto wh = std::make_pair(patch.cols,patch.rows);
        auto iter = rectToIndex.find(wh);
        assert(iter != rectToIndex.end());
        const size_t index = iter->second;
        const auto rect = rectangles[index];
        Rect roi(rect.x,rect.y,rect.w,rect.h);
        packedPatches(roi) = patch;
        rectToIndex.erase(iter);
    }

    imwrite(patches_path, packedPatches);
    
    return 0;
}
