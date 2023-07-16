#include <wrapCreatorGpu.h>

namespace sl {
    namespace wrapCreator {
        void WrapCreatorGpu::getWrapImg(
                const std::vector<cv::Mat> &imgs, cv::cuda::GpuMat &wrapImg,
                cv::cuda::GpuMat &conditionImg,
                cv::cuda::Stream &cvStream, const dim3 block) {
            const int rows = imgs[0].rows;
            const int cols = imgs[0].cols;

            wrapImg.create(rows, cols, CV_32FC1);
            conditionImg.create(rows, cols, CV_32FC1);

            wrapCreator::cudaFunc::getWrapImgSync(imgs, wrapImg,
                                                  conditionImg, cvStream, block);
        }
    }// namespace wrapCreator
}// namespace sl