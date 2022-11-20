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

using nlohmann::json;
using namespace cv;

//
// Simple global patch matching.
// Given:
//   * target image
//   * set of patches + metadata
// Output:
//   Finds the best matching patches in the target image,
//   and if the match is sufficient we output
//   * list of matching patch locations
//   * draw rectangles on matches in target image
//
// Future:
//   * include approx. image pose + camera instrincs to
//     predict patch locations via projection of 3D points.
//

int main(int argc, char *argv[]) {

    if (argc != 6) {
        std::cerr << "usage: " << argv[0]
                  << " image.jgp patches.jpg patches.json matches.jpg matches.json\n";
        exit(1);
    }

    const std::string targetImagePath(argv[1]);
    const std::string patchesImagePath(argv[2]);
    const std::string patchesJsonPath(argv[3]);
    const std::string matchesImagePatch(argv[4]);
    const std::string matchesJsonPath(argv[5]);

    Mat image = imread(targetImagePath, IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Unable to open input image '"
                  << targetImagePath << "'!\n";
        exit(-1);
    }
    Rect imageRect(0, 0, image.cols, image.rows);

    Mat grayImage;
    cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    Mat patchesImage = imread(patchesImagePath, IMREAD_COLOR);
    if (patchesImage.empty()) {
        std::cerr << "Unable to open input patches image '"
                  << patchesImagePath << "'!\n";
        exit(-1);
    }

    Mat patchesGrayImage;
    cvtColor(patchesImage, patchesGrayImage, cv::COLOR_BGR2GRAY);

    std::vector<Rect> packedRects;

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
            assert(pj.find("packed_pos") != pj.end());
            assert(pj["packed_pos"].is_array());
            std::vector<int> ppos = pj["packed_pos"].get<std::vector<int>>();
            assert(pj.find("patch_size") != pj.end());
            assert(pj["patch_size"].is_array());
            std::vector<int> psize = pj["patch_size"].get<std::vector<int>>();
            Rect rect(ppos[0], ppos[1], psize[0], psize[1]);
            packedRects.emplace_back(rect);
        }
    }

    struct MatchInfo {
        double matchVal;
        Point matchLoc;
        MatchInfo(double val, Point loc)
            : matchVal{val}, matchLoc{loc} {}
    };
    std::vector<MatchInfo> matchesInfo;

    double matchesValuesSum = 0;
    double matchesValuesSquaredSum = 0;

    for (auto&& rect : packedRects) {
        Mat patch = patchesGrayImage(rect);
        Mat result;
        matchTemplate(grayImage, patch, result, TM_CCOEFF_NORMED);
        double minVal, maxVal;
        Point minLoc, maxLoc;
        minMaxLoc(result, &minVal, &maxVal, &minLoc,  &maxLoc);
        matchesValuesSum += maxVal;
        matchesValuesSquaredSum += maxVal*maxVal;
        matchesInfo.emplace_back(MatchInfo{maxVal, maxLoc});
        const Scalar color(0, 255, 255); // BGRR
        constexpr int thickness = 2;
        Rect matchRect{maxLoc.x, maxLoc.y, rect.width, rect.height};
        rectangle(image, matchRect, color, thickness);
    }

    const double matchesValuesMean = matchesValuesSum / packedRects.size();
    const double matchesValuesVar = matchesValuesSquaredSum/packedRects.size() - matchesValuesMean*matchesValuesMean;
    const double matchesValuesStdDev = std::sqrt(matchesValuesVar);
    std::cout << "Matches value mean  = " << matchesValuesMean << "\n";
    std::cout << "Matches value stdev = " << matchesValuesStdDev << "\n";

    imwrite(matchesImagePatch, image);

    json matchesData = json::array();
    for (auto&& matches : matchesInfo) {
        json j = json::object();
        j["match_val"] = matches.matchVal;
        j["match_location"] = {matches.matchLoc.x, matches.matchLoc.y};
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
