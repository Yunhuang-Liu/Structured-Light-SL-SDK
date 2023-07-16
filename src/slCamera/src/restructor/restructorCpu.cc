#include <restructorCpu.h>

#include <immintrin.h>
#include <limits>
#include <thread>

namespace sl {
namespace restructor {
RestructorCpu::RestructorCpu(const tool::Info &calibrationInfo)
    : __calibrationInfo(calibrationInfo) {}

void RestructorCpu::restruction(const cv::Mat &leftAbsImg,
                                const cv::Mat &rightAbsImg,
                                const RestructParamater param,
                                cv::Mat &depthImgOut) {
    if (depthImgOut.empty())
        depthImgOut = cv::Mat(leftAbsImg.size(), CV_32FC1, cv::Scalar(0.f));
    else
        depthImgOut.setTo(0);
    getDepthColorMap(leftAbsImg, rightAbsImg, param, depthImgOut);
}

void RestructorCpu::getDepthColorMap(const cv::Mat &leftAbsImg,
                                     const cv::Mat &rightAbsImg,
                                     const RestructParamater param,
                                     cv::Mat &depthImgOut) {
    std::vector<std::thread> tasks(std::thread::hardware_concurrency());
    const int rows = leftAbsImg.rows / tasks.size();
    for (int i = 0; i < tasks.size() - 1; i++) {
        tasks[i] = std::thread(
            &restructor::RestructorCpu::entryDepthColorMap, this,
            std::ref(leftAbsImg), std::ref(rightAbsImg), param,
            std::ref(depthImgOut), cv::Point2i(rows * i, rows * (i + 1)));
    }
    tasks[tasks.size() - 1] =
        std::thread(&restructor::RestructorCpu::entryDepthColorMap, this,
                    std::ref(leftAbsImg), std::ref(rightAbsImg), param,
                    std::ref(depthImgOut),
                    cv::Point2i(rows * (tasks.size() - 1), leftAbsImg.rows));
    for (int i = 0; i < tasks.size(); i++) {
        if (tasks[i].joinable()) {
            tasks[i].join();
        }
    }
}

void RestructorCpu::entryDepthColorMap(const cv::Mat &leftAbsImg,
                                       const cv::Mat &righAbstImg,
                                       const RestructParamater param,
                                       cv::Mat &depthImgOut,
                                       const cv::Point2i region) {
    const cv::Mat &Q = __calibrationInfo.__Q;
    cv::Mat r1Inv = __calibrationInfo.__R1.inv();
    const float f = Q.at<double>(2, 3);
    const float tx = -1.f / Q.at<double>(3, 2);
    const float cxlr = Q.at<double>(3, 3) * tx;
    const float cx = -1.0 * Q.at<double>(0, 3);
    const float cy = -1.0 * Q.at<double>(1, 3);
    const int rows = leftAbsImg.rows;
    const int cols = leftAbsImg.cols;
    for (int i = region.x; i < region.y; ++i) {
        const float *pLeft = leftAbsImg.ptr<float>(i);
        const float *pRight = righAbstImg.ptr<float>(i);
        for (int j = 0; j < cols; ++j) {
            if (0.f >= pLeft[j]) {
                continue;
            }
            float cost = 0;
            int k = 0;
            float minCost = FLT_MAX;
            bool sucessFind = false;
            for (int d = param.__minDisparity; d < param.__maxDisparity; ++d) {
                if (j - d < 0 || j - d > cols - 1) {
                    continue;
                }

                cost = std::abs(pLeft[j] - pRight[j - d]);

                if (sucessFind) {
                    if (cost > minCost) {
                        break;
                    }
                }

                if (cost < minCost) {
                    minCost = cost;
                    k = d;
                    sucessFind = minCost < param.__maximumCost ? true : false;
                }
            }

            if (!sucessFind) {
                continue;
            }

            float dived = pRight[j - k + 1] - pRight[j - k - 1];
            dived = std::abs(dived) < 0.001 ? 0.001 : dived;

            float disparity = k + 2 * (pRight[j - k] - pLeft[j]) / dived;

            if (disparity < param.__minDisparity ||
                disparity > param.__maxDisparity ||
                std::abs(disparity - k) > 1.f) {
                continue;
            }

            cv::Mat recCameraPoints =
                (cv::Mat_<double>(3, 1)
                     << -1.0 * tx * (j - cx) / (disparity - cxlr),
                 -1.0 * tx * (i - cy) / (disparity - cxlr),
                 -1.0 * tx * f / (disparity - cxlr));
            cv::Mat cameraPoints = param.__isMapToPreDepthAxes
                                       ? r1Inv * recCameraPoints
                                       : recCameraPoints;
            cv::Mat result = param.__isMapToColorCamera
                                 ? __calibrationInfo.__Rlc * cameraPoints +
                                       __calibrationInfo.__Tlc
                                 : cameraPoints;

            const float depth = result.at<double>(2, 0);
            if (depth < param.__minDepth || depth > param.__maxDepth)
                continue;

            if (param.__isMapToPreDepthAxes) {
                cv::Mat mapPicture = param.__isMapToColorCamera
                                         ? __calibrationInfo.__M3 * result
                                         : __calibrationInfo.__M1 * result;
                int x_maped = std::round(mapPicture.at<double>(0, 0) /
                                         mapPicture.at<double>(2, 0));
                int y_maped = std::round(mapPicture.at<double>(1, 0) /
                                         mapPicture.at<double>(2, 0));

                if ((0 > x_maped) || (y_maped > rows - 1) || (0 > y_maped) ||
                    (x_maped > cols - 1))
                    continue;

                std::lock_guard<std::mutex> lock(__mutexMap);
                depthImgOut.ptr<float>(y_maped)[x_maped] = depth;
            } else {
                depthImgOut.ptr<float>(i)[j] = depth;
            }
        }
    }
}
} // namespace restructor
} // namespace sl
