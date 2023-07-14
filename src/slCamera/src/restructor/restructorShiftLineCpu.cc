#include <restructorShiftLineCpu.h>

namespace sl {
namespace restructor {
RestructorShiftLineCpu::RestructorShiftLineCpu(const tool::Info &calibrationInfo)
    : __calibrationInfo(calibrationInfo) {}

void RestructorShiftLineCpu::restruction(const cv::Mat &leftCodeImg,
                                const cv::Mat &rightCodeImg,
                                const RestructParamater param,
                                cv::Mat &depthImgOut) {
    if (depthImgOut.empty())
        depthImgOut = cv::Mat( leftCodeImg.size(), CV_32FC1, cv::Scalar(0.f));
    else
        depthImgOut.setTo(0);
    getDepthColorMap(leftCodeImg, rightCodeImg, param, depthImgOut);
}

void RestructorShiftLineCpu::getDepthColorMap(const cv::Mat &leftCodeImg,
                                     const cv::Mat &rightCodeImg,
                                     const RestructParamater param,
                                     cv::Mat &depthImgOut) {
    std::vector<std::thread> tasks(std::thread::hardware_concurrency());
    const int rows = leftCodeImg.rows / tasks.size();
    for (int i = 0; i < tasks.size() - 1; i++) {
        tasks[i] = std::thread(
            &restructor::RestructorShiftLineCpu::entryDepthColorMap, this,
            std::ref(leftCodeImg), std::ref(rightCodeImg), param,
            std::ref(depthImgOut), cv::Point2i(rows * i, rows * (i + 1)));
    }
    tasks[tasks.size() - 1] =
        std::thread(&restructor::RestructorShiftLineCpu::entryDepthColorMap, this,
                    std::ref(leftCodeImg), std::ref(rightCodeImg), param,
                    std::ref(depthImgOut),
                    cv::Point2i(rows * (tasks.size() - 1), leftCodeImg.rows));
    for (int i = 0; i < tasks.size(); i++) {
        if (tasks[i].joinable()) {
            tasks[i].join();
        }
    }
}

void RestructorShiftLineCpu::entryDepthColorMap(const cv::Mat &leftCodeImg,
                                       const cv::Mat &rightCodeImg,
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
    const int rows = leftCodeImg.rows;
    const int cols = leftCodeImg.cols;
    for (int i = region.x; i < region.y; ++i) {
        const cv::Vec4f *pLeft = leftCodeImg.ptr<cv::Vec4f>(i);
        const cv::Vec4f *pRight = rightCodeImg.ptr<cv::Vec4f>(i);
        for (int j = 0; j < cols; ++j) {
            if (0.f >= pLeft[j][0]) {
                continue;
            }

            bool sucessFind = false;
            float disparity = FLT_MAX;
            for (int d = param.__minDisparity; d < param.__maxDisparity; ++d) {
                if (j - d < 0 || j - d > cols - 1 || 0.f >= pRight[j - d][0]) {
                    continue;
                }

                if(pLeft[j][1] == pRight[j - d][1] && pLeft[j][2] == pRight[j - d][2] && pLeft[j][3] == pRight[j - d][3]) {
                    sucessFind = true;
                    disparity = pLeft[j][0] - pRight[j - d][0];
                    break;
                }
            }

            if (!sucessFind) {
                continue;
            }

            if (disparity < param.__minDisparity ||
                disparity > param.__maxDisparity) {
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
