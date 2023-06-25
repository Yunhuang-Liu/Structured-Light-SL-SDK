#include <cudaTypeDef.cuh>
#include <cuda_runtime_api.h>

#include <phaseSolver.h>
#include <typeDef.h>


namespace sl {
namespace phaseSolver {

namespace cudaFunc {
__global__ void solvePhaseCuda(IN const cv::cuda::PtrStep<uchar> imgs,
                               IN const int imgsSize,
                               IN const int phaseShifteTime, IN const int rows,
                               IN const int cols, IN const float sncThreshold,
                               IN cv::cuda::PtrStep<float> wrapImg,
                               IN cv::cuda::PtrStep<float> conditionImg,
                               IN cv::cuda::PtrStep<float> unwrapImg) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > cols - 1 || x < 0 || y < 0 || y > rows - 1) {
        return;
    }
    // 调制度、相移偏移量、包裹正弦部分、包裹余弦部分
    float snc = 0.f, curShift = 0.f, curSin = 0.f, curCos = 0.f;
    for (size_t i = 0; i < phaseShifteTime; ++i) {
        curShift = i * CV_2PI / phaseShifteTime;
        snc += imgs.ptr(y)[imgsSize * x + i];
        curSin += imgs.ptr(y)[imgsSize * x + i] * sin(curShift);
        curCos += imgs.ptr(y)[imgsSize * x + i] * cos(curShift);
    }

    snc /= phaseShifteTime;
    conditionImg.ptr(y)[x] = snc;

    // 计算包裹相位
    const float wrapVal = -1.f * cuda::std::atan2(curSin, curCos);
    wrapImg.ptr(y)[x] = wrapVal;

    if (snc < sncThreshold) {
        unwrapImg.ptr(y)[x] = 0.f;
        return;
    }
    // 计算计算绝对相位
    int grayCodeK1 = 0, grayCodeK2 = 0, preGrayBit = 0;
    for (size_t i = phaseShifteTime; i < imgsSize; ++i) {
        const int curGrayBit = imgs.ptr(y)[imgsSize * x + i] < snc ? 0 : 1;
        preGrayBit = (i == phaseShifteTime) ? curGrayBit ^ 0 : curGrayBit ^ preGrayBit;
        grayCodeK2 +=
            preGrayBit * cuda::std::pow(2, imgsSize - i - 1);

        if (i != imgsSize - 1) {
            grayCodeK1 += preGrayBit *
                          cuda::std::pow(2, imgsSize - i - 2);
        }
    }

    grayCodeK2 = (grayCodeK2 + 1) / 2;
    if (wrapVal > -CV_PI / 2 && wrapVal < CV_PI / 2) {
        unwrapImg.ptr(y)[x] = wrapVal + CV_2PI * grayCodeK1 + CV_PI;
    }
    else if (wrapVal <= -CV_PI / 2) {
        unwrapImg.ptr(y)[x] = wrapVal + CV_2PI * grayCodeK2 + CV_PI;
    }
    else {
        unwrapImg.ptr(y)[x] = wrapVal + CV_2PI * (grayCodeK2 - 1) + CV_PI;
    }
}

void solvePhase(IN const std::vector<cv::Mat> &imgs,
                OUT PhaseSolverGroupDataDevice &groupData,
                IN const float sncThreshold, IN const int shiftTime,
                IN cv::cuda::Stream &cvStream, IN const dim3 block) {
    CV_Assert(imgs.size() != 0);
    const int rows = imgs[0].rows;
    const int cols = imgs[0].cols;
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    cv::Mat mergeImg;
    cv::cuda::GpuMat deviceImgs;
    cv::merge(imgs, mergeImg);
    deviceImgs.upload(mergeImg, cvStream);

    solvePhaseCuda<<<grid, block, 0, stream>>>(
        deviceImgs, imgs.size(), shiftTime, rows, cols, sncThreshold,
        groupData.__wrapMap, groupData.__textureMap, groupData.__unwrapMap);
}
} // namespace cudaFunc
} // namespace phaseSolver
} // namespace sl