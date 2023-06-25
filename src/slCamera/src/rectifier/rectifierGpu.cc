#include <rectifierGpu.h>

namespace sl {
    namespace rectifier {
        RectifierGpu::RectifierGpu() {
        }

        RectifierGpu::RectifierGpu(const tool::Info &info) {
            __imgSize = cv::Size(info.__S.ptr<double>(0)[0], info.__S.ptr<double>(1)[0]);
            cv::Mat mapLx, mapLy, mapRx, mapRy;
            cv::initUndistortRectifyMap(info.__M1, info.__D1, info.__R1, info.__P1, __imgSize, CV_32FC1, mapLx, mapLy);
            cv::initUndistortRectifyMap(info.__M2, info.__D2, info.__R2, info.__P2, __imgSize, CV_32FC1, mapRx, mapRy);
            __mapLx.upload(mapLx);
            __mapLy.upload(mapLy);
            __mapRx.upload(mapRx);
            __mapRy.upload(mapRy);
        }

        void RectifierGpu::initialize(const tool::Info &info) {
            __imgSize = cv::Size(info.__S.ptr<double>(0)[0], info.__S.ptr<double>(1)[0]);
            cv::Mat mapLx, mapLy, mapRx, mapRy;
            cv::initUndistortRectifyMap(info.__M1, info.__D1, info.__R1, info.__P1, __imgSize, CV_32FC1, mapLx, mapLy);
            cv::initUndistortRectifyMap(info.__M2, info.__D2, info.__R2, info.__P2, __imgSize, CV_32FC1, mapRx, mapRy);
            __mapLx.upload(mapLx);
            __mapLy.upload(mapLy);
            __mapRx.upload(mapRx);
            __mapRy.upload(mapRy);
        }

        void RectifierGpu::remapImg(const cv::cuda::GpuMat& imgInput, cv::cuda::GpuMat &imgOutput,
                                    const bool isLeft, cv::cuda::Stream &cvStream) {
            //cv::cuda::GpuMat imgDev;
            //imgDev.upload(imgInput, cvStream);
            if (isLeft)
                cv::cuda::remap(imgInput, imgOutput, __mapLx, __mapLy, cv::INTER_LINEAR,
                                0, cv::Scalar(), cvStream);
            else
                cv::cuda::remap(imgInput, imgOutput, __mapRx, __mapRy, cv::INTER_LINEAR,
                                0, cv::Scalar(), cvStream);
        }
    }// namespace rectifier
}// namespace sl