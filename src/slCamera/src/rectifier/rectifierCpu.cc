#include <rectifierCpu.h>

namespace sl {
    namespace rectifier {
        RectifierCpu::RectifierCpu() {
        }

        RectifierCpu::RectifierCpu(const tool::Info &info) {
            __imgSize = cv::Size(info.__S.ptr<double>(0)[0], info.__S.ptr<double>(1)[0]);
            cv::initUndistortRectifyMap(info.__M1, info.__D1, info.__R1, info.__P1, __imgSize, CV_32FC1, __mapLx, __mapLy);
            cv::initUndistortRectifyMap(info.__M2, info.__D2, info.__R2, info.__P2, __imgSize, CV_32FC1, __mapRx, __mapRy);
        }

        void RectifierCpu::initialize(const tool::Info &info) {
            __imgSize = cv::Size(info.__S.ptr<double>(0)[0], info.__S.ptr<double>(1)[0]);
            cv::initUndistortRectifyMap(info.__M1, info.__D1, info.__R1, info.__P1, __imgSize, CV_32FC1, __mapLx, __mapLy);
            cv::initUndistortRectifyMap(info.__M2, info.__D2, info.__R2, info.__P2, __imgSize, CV_32FC1, __mapRx, __mapRy);
        }

        void RectifierCpu::remapImg(const cv::Mat &imgInput, cv::Mat &imgOutput, const bool isLeft) {
            if (isLeft)
                cv::remap(imgInput, imgOutput, __mapLx, __mapLy, cv::INTER_LINEAR);
            else
                cv::remap(imgInput, imgOutput, __mapRx, __mapRy, cv::INTER_LINEAR);
        }
    }// namespace rectifier
}// namespace sl