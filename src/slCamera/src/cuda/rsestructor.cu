#include <cudaTypeDef.cuh>

#include <atomic>

#include <restructor.h>

namespace sl {
namespace restructor {

namespace cudaFunc {
__global__ void matchAndTriangulateCUDA(
    cv::cuda::PtrStep<float> leftImg, cv::cuda::PtrStep<float> rightImg,
    const int rows, const int cols, const int minDisparity,
    const int maxDisparity, const float minDepth, const float maxDepth,
    const float maximumCost, const Eigen::Matrix4f Q, const Eigen::Matrix3f M1,
    const Eigen::Matrix3f M3, const Eigen::Matrix3f R, const Eigen::Vector3f T,
    const Eigen::Matrix3f R1_inv, cv::cuda::PtrStep<float> mapDepth,
    const bool isMapToPreAxes, const bool isMapToColorCamera) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        if (0 >= leftImg.ptr(y)[x]) {
            return;
        }
        const float f = Q(2, 3);
        const float tx = -1.0 / Q(3, 2);
        const float cxlr = Q(3, 3) * tx;
        const float cx = -1.0 * Q(0, 3);
        const float cy = -1.0 * Q(1, 3);
        float cost = 0;
        int k = 0;
        float minCost = FLT_MAX;
        bool sucessFind = false;
        for (int d = minDisparity; d < maxDisparity; ++d) {
            if (x - d < 0 || x - d > cols - 1) {
                continue;
            }

            cost = std::abs(leftImg.ptr(y)[x] - rightImg.ptr(y)[x - d]);

            if(sucessFind) {
                if(cost > minCost) {
                    break;
                }
            }

            if (cost < minCost) {
                minCost = cost;
                k = d;
                sucessFind = minCost < maximumCost ? true : false;
            }
        }

        if(!sucessFind) {
            return;
        }

        float dived = rightImg.ptr(y)[x - k + 1] - rightImg.ptr(y)[x - k - 1];

        if (std::abs(dived) < 0.001) {
            dived = 0.001;
        }

        float disparity =
            k + 2 * (rightImg.ptr(y)[x - k] - leftImg.ptr(y)[x]) / dived;

        if (disparity < minDisparity || disparity > maxDisparity || std::abs(disparity - k) > 1.f) {
            return;
        }

        Eigen::Vector3f vertex;
        vertex(0, 0) = -1.0f * tx * (x - cx) / (disparity - cxlr);
        vertex(1, 0) = -1.0f * tx * (y - cy) / (disparity - cxlr);
        vertex(2, 0) = -1.0f * tx * f / (disparity - cxlr);

        const Eigen::Vector3f depthVertex =
            isMapToPreAxes ? R1_inv * vertex : vertex;
        const Eigen::Vector3f colorVertex =
            isMapToColorCamera ? R * (R1_inv * vertex) + T : depthVertex;

        if (isMapToPreAxes) {
            const Eigen::Vector3f imgMapped =
                isMapToColorCamera ? M3 * colorVertex : M1 * colorVertex;
            const int x_maped = imgMapped(0, 0) / imgMapped(2, 0);
            const int y_maped = imgMapped(1, 0) / imgMapped(2, 0);
            const float depthMaped = colorVertex(2, 0);
            if (x_maped < cols && y_maped < rows && 0 <= x_maped &&
                0 <= y_maped) {
                if (depthMaped < minDepth || depthMaped > maxDepth) {
                    atomicExch(&mapDepth.ptr(y_maped)[x_maped], 0);
                }
                else {
                    atomicExch(&mapDepth.ptr(y_maped)[x_maped], depthMaped);
                }
            }
        } 
        else {
            mapDepth.ptr(y)[x] = colorVertex(2, 0);
        }
    }
}

void getDepthMap(const cv::cuda::GpuMat &leftImg,
                 const cv::cuda::GpuMat &rightImg,
                 const RestructParamater param, const Eigen::Matrix4f &Q,
                 const Eigen::Matrix3f &M1, const Eigen::Matrix3f &M3,
                 const Eigen::Matrix3f &R, const Eigen::Vector3f &T,
                 const Eigen::Matrix3f &R1Inv, cv::cuda::GpuMat &depthMap,
                 cv::cuda::Stream &cvStream, const dim3 block) {
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
    dim3 grid((leftImg.cols + block.x - 1) / block.x,
              (leftImg.rows + block.y - 1) / block.y, 1);
    matchAndTriangulateCUDA<<<grid, block, 0, stream>>>(
        leftImg, rightImg, leftImg.rows, leftImg.cols, param.__minDisparity,
        param.__maxDisparity, param.__minDepth, param.__maxDepth,
        param.__maximumCost, Q, M1, M3, R, T, R1Inv, depthMap,
        param.__isMapToPreDepthAxes, param.__isMapToColorCamera);
}
} // namespace cudaFunc
} // namespace restructor
} // namespace sl
