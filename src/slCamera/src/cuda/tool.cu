#include <cudaTypeDef.cuh>

#include <matrixsInfo.h>

namespace sl {
namespace tool {
namespace cudaFunc {
/**
 * @brief               全图像相位高度映射（CUDA加速优化核函数）
 *
 * @param phase         相位图
 * @param rows          行数
 * @param cols          列数
 * @param intrinsic     内参
 * @param coefficient   八参数
 * @param minDepth      最小深度
 * @param maxDepth      最大深度
 * @param depth         深度图
 */
__global__ void phaseHeightMapEigCoe_Device(
    IN const cv::cuda::PtrStep<float> phase, IN const int rows,
    IN const int cols, IN const Eigen::Matrix3f intrinsic,
    IN const Eigen::Vector<float, 8> coefficient, IN const float minDepth,
    IN const float maxDepth, OUT cv::cuda::PtrStep<float> depth) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x > cols - 1 || y > rows - 1)
        return;

    if (phase.ptr(y)[x] == -5.f) {
        depth.ptr(y)[x] = 0.f;
        return;
    }

    Eigen::Matrix3f mapL;
    Eigen::Vector3f mapR, cameraPoint;

    mapL(0, 0) = intrinsic(0, 0);
    mapL(0, 1) = 0;
    mapL(0, 2) = intrinsic(0, 2) - x;
    mapL(1, 0) = 0;
    mapL(1, 1) = intrinsic(1, 1);
    mapL(1, 2) = intrinsic(1, 2) - y;
    mapL(2, 0) = coefficient(0, 0) - coefficient(4, 0) * phase.ptr(y)[x];
    mapL(2, 1) = coefficient(1, 0) - coefficient(5, 0) * phase.ptr(y)[x];
    mapL(2, 2) = coefficient(2, 0) - coefficient(6, 0) * phase.ptr(y)[x];

    mapR(0, 0) = 0;
    mapR(1, 0) = 0;
    mapR(2, 0) = coefficient(7, 0) * phase.ptr(y)[x] - coefficient(3, 0);

    cameraPoint = mapL.inverse() * mapR;
    depth.ptr(y)[x] = cameraPoint.z();
}

/**
 * @brief               全图像相位高度映射（CUDA加速优化核函数）
 *
 * @param depth         深度图
 * @param textureSrc    纹理相机采集的纹理图
 * @param rows          行数
 * @param cols          列数
 * @param intrinsicInvD 深度相机内参矩阵逆矩阵
 * @param intrinsicT    纹理相机内参
 * @param rotateDToT    深度相机到纹理相机的旋转矩阵
 * @param translateDtoT 深度相机到纹理相机的平移矩阵
 * @param textureMapped 映射到深度相机下的纹理
 */
__global__ void reverseMappingTexture_Device(
    IN const cv::cuda::PtrStep<float> depth,
    IN const cv::cuda::PtrStep<uchar3> textureSrc, IN const int rows,
    IN const int cols, IN const Eigen::Matrix3f intrinsicInvD,
    IN const Eigen::Matrix3f intrinsicT, IN const Eigen::Matrix3f rotateDToT,
    IN const Eigen::Vector3f translateDtoT,
    OUT cv::cuda::PtrStep<uchar3> textureMapped) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x > cols - 1 || y > rows - 1)
        return;

    if (depth.ptr(y)[x] == 0.f)
        return;

    Eigen::Vector3f imgPoint(x * depth.ptr(y)[x], y * depth.ptr(y)[x],
                             depth.ptr(y)[x]);
    Eigen::Vector3f texturePoint =
        intrinsicT * (rotateDToT * (intrinsicInvD * imgPoint) + translateDtoT);

    const int xTexture = texturePoint(0, 0) / texturePoint(2, 0);
    const int yTexture = texturePoint(1, 0) / texturePoint(2, 0);

    if (xTexture < 0 || xTexture > cols - 1 || yTexture < 0 ||
        yTexture > rows - 1)
        return;

    textureMapped.ptr(y)[x] = textureSrc.ptr(yTexture)[xTexture];
}
/**
 * @brief               计算纹理图片（CUDA加速优化核函数）
 *
 * @param imgs          纹理合并图
 * @param imgsSize      图片张数
 * @param rows          行数
 * @param cols          列数
 * @param texture       纹理图
 */
__global__ void averageTexture_Device(IN const cv::cuda::PtrStep<uchar> imgs,
                                      IN const int imgsSize, IN const int rows,
                                      IN const int cols,
                                      OUT cv::cuda::PtrStep<uchar> texture) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x > cols - 1 || y > rows - 1)
        return;

    for (size_t i = 0; i < imgsSize; ++i) {
        texture.ptr(y)[3 * x + 0] += imgs.ptr(y)[3 * imgsSize * x + 3 * i + 0];
        texture.ptr(y)[3 * x + 1] += imgs.ptr(y)[3 * imgsSize * x + 3 * i + 1];
        texture.ptr(y)[3 * x + 2] += imgs.ptr(y)[3 * imgsSize * x + 3 * i + 2];
    }
}
/**
 * @brief               过滤相位（CUDA加速优化核函数）
 *
 * @param absPhase      绝对相位
 * @param rows          行数
 * @param cols          列数
 * @param maxTollerance 最大不同量
 * @param kernel        核大小
 * @param out           过滤后的绝对相位图
 */
__global__ void filterPhase_Device(IN const cv::cuda::PtrStep<float> absPhase,
                                   IN const int rows, IN const int cols,
                                   IN const float maxTollerance,
                                   IN const int kernel,
                                   OUT cv::cuda::PtrStep<float> out) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < kernel / 2 || x > cols - kernel / 2 || y < kernel / 2 ||
        y > rows - kernel / 2) {
        return;
    }

    int diffCount = 0;
    const float diffTolerrance = maxTollerance;
    for (int d = -kernel / 2; d < kernel / 2; ++d) {
        for (int k = -kernel / 2; k < kernel / 2; ++k) {
            diffCount = std::abs(absPhase.ptr(y + d)[x + k] -
                                 absPhase.ptr(y)[x]) > diffTolerrance
                            ? diffCount + 1
                            : diffCount;
        }
    }

    out.ptr(y)[x] = diffCount > (kernel / 2 + 1) * (kernel / 2 + 1)
                        ? 0.f
                        : absPhase.ptr(y)[x];
}

void phaseHeightMapEigCoe(
    const cv::cuda::GpuMat &phase, const Eigen::Matrix3f &intrinsic,
    const Eigen::Vector<float, 8> &coefficient, const float minDepth,
    const float maxDepth, cv::cuda::GpuMat &depth,
    const dim3 block,
    cv::cuda::Stream &cvStream) {

    CV_Assert(!phase.empty() && phase.type() == CV_32FC1);

    depth.create(phase.rows, phase.cols, CV_32FC1);
    depth.setTo(0.f);

    const int rows = phase.rows;
    const int cols = phase.cols;

    const dim3 grid((cols + block.x - 1) / block.x,
                    (rows + block.y - 1) / block.y);
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
    phaseHeightMapEigCoe_Device<<<grid, block, 0, stream>>>(
        phase, phase.rows, phase.cols, intrinsic, coefficient, minDepth,
        maxDepth, depth);
}

void reverseMappingTexture(
    const cv::cuda::GpuMat &depth, const cv::cuda::GpuMat &textureSrc,
    const Eigen::Matrix3f &intrinsicInvD, const Eigen::Matrix3f &intrinsicT,
    const Eigen::Matrix3f &rotateDToT, const Eigen::Vector3f &translateDtoT,
    cv::cuda::GpuMat &textureMapped, const dim3 block,
    cv::cuda::Stream &cvStream) {
    CV_Assert(depth.type() == CV_32FC1 && !depth.empty() &&
              textureSrc.type() == CV_8UC3 && !textureSrc.empty());

    textureMapped.create(textureSrc.size(), CV_8UC3);
    textureMapped.setTo(cv::Scalar(0.f, 0.f, 0.f));

    const int rows = textureSrc.rows;
    const int cols = textureSrc.cols;
    const dim3 grid((cols + block.x - 1) / block.x,
                    (rows + block.y - 1) / block.y);
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
    reverseMappingTexture_Device<<<grid, block, 0, stream>>>(
        depth, textureSrc, rows, cols, intrinsicInvD, intrinsicT, rotateDToT,
        translateDtoT, textureMapped);
}

void averageTexture(const std::vector<cv::Mat> &imgs, cv::cuda::GpuMat &texture,
                    const dim3 block,
                    cv::cuda::Stream &cvStream) {

    texture.create(imgs[0].size(), CV_8UC3);
    texture.setTo(cv::Scalar(0, 0, 0));

    const int rows = texture.rows;
    const int cols = texture.cols;
    const dim3 grid((cols + block.x - 1) / block.x,
                    (rows + block.y - 1) / block.y);
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);

    cv::Mat mergeImg;
    cv::cuda::GpuMat deviceImgs;
    cv::merge(imgs, mergeImg);
    deviceImgs.upload(mergeImg, cvStream);

    averageTexture_Device<<<grid, block, 0, stream>>>(deviceImgs, imgs.size(),
                                                      rows, cols, texture);
}

void filterPhase(IN const cv::cuda::GpuMat &absPhase, OUT cv::cuda::GpuMat &out,
                 IN const float maxTollerance, IN const int kernel,
                 IN const dim3 block,
                 IN cv::cuda::Stream &cvStream) {
    out.create(absPhase.size(), CV_32FC1);
    out.setTo(0.f);

    const int rows = absPhase.rows;
    const int cols = absPhase.cols;
    const dim3 grid((cols + block.x - 1) / block.x,
                    (rows + block.y - 1) / block.y);
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);

    filterPhase_Device<<<grid, block, 0, stream>>>(absPhase, rows, cols,
                                                   maxTollerance, kernel, out);
}

} // namespace cudaFunc
} // namespace tool
} // namespace sl