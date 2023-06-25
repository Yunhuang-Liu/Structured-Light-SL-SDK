#include <cudaTypeDef.cuh>

#include <typeDef.h>

namespace sl {
namespace wrapCreator {
namespace cudaFunc {
__global__ void solveWrapCuda(IN const cv::cuda::PtrStep<uchar> imgs,
                              IN const int imgsSize, IN const int rows,
                              IN const int cols, 
                              IN cv::cuda::PtrStep<float> wrapImg,
                              IN cv::cuda::PtrStep<float> conditionImg) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > cols - 1 || x < 0 || y < 0 || y > rows - 1) {
        return;
    }
    // 调制度、相移偏移量、包裹正弦部分、包裹余弦部分
    float snc = 0.f, curShift = 0.f, curSin = 0.f, curCos = 0.f;
    for (size_t i = 0; i < imgsSize; ++i) {
        curShift = i * CV_2PI / imgsSize;
        snc += imgs.ptr(y)[imgsSize * x + i];
        curSin += imgs.ptr(y)[imgsSize * x + i] * sin(curShift);
        curCos += imgs.ptr(y)[imgsSize * x + i] * cos(curShift);
    }
    snc /= imgsSize;

    // 计算包裹相位
    const float wrapVal = -1.f * cuda::std::atan2(curSin, curCos);
    
    wrapImg.ptr(y)[x] = wrapVal;
    conditionImg.ptr(y)[x] = snc;
}

void getWrapImgSync(IN const std::vector<cv::Mat> &imgs,
                    OUT cv::cuda::GpuMat &wrapImg, OUT cv::cuda::GpuMat &conditionImg,
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

    solveWrapCuda<<<grid, block, 0, stream>>>(
        deviceImgs, imgs.size(), rows, cols,
        wrapImg, conditionImg);
}
} // namespace cudaFunc
} // namespace wrapCreator
} // namespace sl