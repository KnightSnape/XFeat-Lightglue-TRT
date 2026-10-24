#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include "utils.h"
#include "xfeat.h"
#include "lightglue.h"

std::vector<MatchPoint> matches;
std::vector<cv::KeyPoint> k1,k2;
std::vector<cv::Point2f> kpts1, kpts2;
std::vector<float> k1_v, k2_v, f1_v, f2_v;
torch::Tensor feats1, keypoints1, heatmap1, idx1;
torch::Tensor feats2, keypoints2, heatmap2, idx2;

const std::filesystem::path project_root =
    std::filesystem::path(__FILE__).parent_path().parent_path();
std::string image1_path = (project_root / "assets/kitti_00_left_000100.png").string();
std::string image2_path = (project_root / "assets/kitti_00_right_000100.png").string();
std::string result_path = (project_root / "assets/kitti_00_000100_matches.png").string();
std::string config_path = (project_root / "config/xfeat_lightglue.yaml").string();
std::string xfeat_engine_path = (project_root / "weights/xfeat_1_352_1216_fp16.engine").string();
std::string glue_engine_path = (project_root / "weights/lightglue_L6_1_352_1216_fp16.engine").string();
// Xfeat 特征提取模块
XFeat xfeat(config_path, xfeat_engine_path);
// LIghtglue 特征匹配模块
Lightglue lightglue(config_path, glue_engine_path);


void feature_extract_match(const cv::Mat& img1, const cv::Mat& img2){
    // auto start = std::chrono::high_resolution_clock::now();
    xfeat.detectAndCompute(img1, keypoints1, feats1, heatmap1); // 提取图1 的特征点和描述子
    xfeat.detectAndCompute(img2, keypoints2, feats2, heatmap2); // 提取图2 的特征点和描述子

    TensorToVectorKeypoints(keypoints1, k1_v);
    TensorToVectorKeypoints(keypoints2, k2_v);
    TensorToVectorKeypoints(feats1, f1_v);
    TensorToVectorKeypoints(feats2, f2_v);
    // 匹配
    lightglue.matching(k1_v, k2_v, f1_v, f2_v, matches);
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // std::cout << "Matching done, time: " << duration << std::endl;

    kpts1 = floatVecToPoints(k1_v);
    kpts2 = floatVecToPoints(k2_v);
}

void draw_match_result(const cv::Mat& img1, const cv::Mat& img2, std::vector<MatchPoint> draw_matches){
    // 估计仿射变换矩阵（只包含旋转、缩放和平移，不包括投影）
    cv::Mat inliers;
    std::vector<cv::Point2f> kpts1_sort, kpts2_sort;
    for (const auto& match : draw_matches) {
        kpts1_sort.push_back(kpts1[match.idx1]);  // idx1是kpts1中点的索引
        kpts2_sort.push_back(kpts2[match.idx2]);  // idx2是kpts2中点的索引
    }
    if (kpts1_sort.size() < 4) {
        throw std::runtime_error("Too few matches to estimate a homography");
    }
    cv::Mat homography = cv::findHomography(kpts1_sort, kpts2_sort, cv::RANSAC, 3, inliers);
    if (homography.empty()) {
        throw std::runtime_error("Unable to estimate a homography from matches");
    }

    // 绘制封闭曲线（绿色线段）
    cv::Mat img1_with_corners;
    cv::Mat img2_with_corners;
    cv::cvtColor(img1, img1_with_corners, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, img2_with_corners, cv::COLOR_GRAY2BGR);

    // 将角点转换到第二幅图像空间
    std::vector<cv::Point2f> warped_corners;
    int h = img1_with_corners.rows;
    int w = img1_with_corners.cols;
    std::vector<cv::Point2f> corners_img1 = {
    {0.f,      0.f},
    {w - 1.f,  0.f},
    {w - 1.f,  h - 1.f},
    {0.f,      h - 1.f}
    };
    cv::perspectiveTransform(corners_img1, warped_corners, homography);

    for (size_t i = 0; i < warped_corners.size(); ++i) {
        cv::Point start_point = warped_corners[(i + warped_corners.size() - 1) % warped_corners.size()];
        cv::Point end_point   = warped_corners[i];
        cv::line(img2_with_corners, start_point, end_point, cv::Scalar(0, 255, 0), 4);
    }

    // 拼接图像
    cv::Mat result_img;
    cv::hconcat(img1_with_corners, img2_with_corners, result_img);
    int offsetX = w;
    for (const auto& match : draw_matches) {
        if (match.idx1 >= kpts1.size() || match.idx2 >= kpts2.size()) continue;
        const auto& pt1 = kpts1[match.idx1];
        const auto& pt2 = kpts2[match.idx2];

        // 生成文本：编号 + 置信度（保留两位小数）
        std::ostringstream oss;
        oss << match.idx1 << " (" << std::fixed << std::setprecision(2) << match.score << ")";
        // 在点的右上方写上索引
        cv::putText(result_img, oss.str(), pt1 + cv::Point2f(5, -5),  // 位置偏移
                    cv::FONT_HERSHEY_SIMPLEX, 0.4,            // 字体和缩放
                    cv::Scalar(0, 0, 255), 1);                // 红色字体


        float s = std::min(std::max(match.score, 0.f), 1.f);
        cv::Scalar color(0, 255 * s, 255 * (1 - s));  // Green → Red
        
        cv::Point2f pt2_shifted = pt2 + cv::Point2f(offsetX, 0);
        cv::circle(result_img, pt1, 3, color, -1);
        cv::circle(result_img, pt2_shifted, 3, color, -1);
        cv::line(result_img, pt1, pt2_shifted, color, 1);
    }
    if (!cv::imwrite(result_path, result_img)) {
        throw std::runtime_error("Unable to write match result: " + result_path);
    }
}


int main(int argc, char* argv[])
{
    // Synchronize to ensure all operations are complete before starting the timer
    torch::cuda::synchronize();
    const cv::Mat img1 = cv::imread(image1_path, cv::IMREAD_GRAYSCALE);
    const cv::Mat img2 = cv::imread(image2_path, cv::IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        std::cerr << "Unable to load KITTI stereo images." << std::endl;
        return 1;
    }

    const auto cold_start = std::chrono::steady_clock::now();
    feature_extract_match(img1, img2);
    torch::cuda::synchronize();
    const double cold_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - cold_start).count();

    constexpr int warmup_iterations = 5;
    constexpr int benchmark_iterations = 20;
    for (int i = 0; i < warmup_iterations; ++i) {
        feature_extract_match(img1, img2);
    }

    std::vector<double> timings;
    timings.reserve(benchmark_iterations);
    for (int i = 0; i < benchmark_iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        feature_extract_match(img1, img2);
        const auto end = std::chrono::steady_clock::now();
        timings.push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
    }

    std::sort(timings.begin(), timings.end());
    const double total_ms = std::accumulate(timings.begin(), timings.end(), 0.0);
    const double mean_ms = total_ms / timings.size();
    const double median_ms = timings[timings.size() / 2];
    const double p95_ms = timings[static_cast<size_t>(
        0.95 * static_cast<double>(timings.size() - 1))];
    draw_match_result(img1, img2, matches);

    std::cout << "KITTI stereo match: left_keypoints=" << keypoints1.size(0)
              << " right_keypoints=" << keypoints2.size(0)
              << " accepted_matches=" << matches.size()
              << " cold_ms=" << cold_ms
              << " mean_ms=" << mean_ms
              << " median_ms=" << median_ms
              << " p95_ms=" << p95_ms
              << " min_ms=" << timings.front()
              << " max_ms=" << timings.back()
              << " fps=" << 1000.0 / mean_ms
              << " result=" << result_path << std::endl;
    return 0;
}
