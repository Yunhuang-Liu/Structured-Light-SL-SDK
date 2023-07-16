/**
 * @file phaseSolver.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  解相器
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */

#ifndef __PHASESOLVER_PHASESOLVER_H_
#define __PHASESOLVER_PHASESOLVER_H_

#ifdef __WITH_CUDA__
//#include <cuda/std/cmath>
#include <cuda/std/functional>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudalegacy.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#endif // !__WITH_CUDA__


#include <thread>
#include <vector>

#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <typeDef.h>

/** @brief 结构光库 */
namespace sl {
/** @brief 解相库 */
namespace phaseSolver {
/** @brief 主机端解相数据 */
struct PhaseSolverGroupDataHost {
    cv::Mat __textureMap; //纹理(CV_32FC1)
    cv::Mat __wrapMap;    //包裹相位(CV_32FC1)
    cv::Mat __unwrapMap;  //绝对相位(CV_32FC1)
};
/** @brief 设备端解相数据 */
struct PhaseSolverGroupDataDevice {
    cv::cuda::GpuMat __textureMap;//纹理(CV_32FC1)
    cv::cuda::GpuMat __wrapMap;   //包裹相位(CV_32FC1)
    cv::cuda::GpuMat __unwrapMap; //绝对相位(CV_32FC1)
};
#ifdef __WITH_CUDA__
/** @brief cuda函数库 */
namespace cudaFunc {
void solvePhase(IN const std::vector<cv::Mat> &imgs,
                OUT PhaseSolverGroupDataDevice &groupData,
                IN const float sncThreshold, 
                IN const int shiftTime = 4,
                IN cv::cuda::Stream &stream = cv::cuda::Stream::Null(),
                IN const dim3 block = dim3(32, 8));
} // namespace cudaFunc
#endif // !__WITH_CUDA__

/** @brief 相位求解器 */
class PhaseSolver {
  public:
    /**
     * @brief 析构函数
     */
    virtual ~PhaseSolver() {}
    /**
     * @brief 解相
     *
     * @param imgs 图像(CV_8UC1 || CV_32FC1)
     * @param groupData 求解结果
     * @param sncThreshold 信噪比阈值
     * @param shiftTime 相移步数
     */
    virtual void solve(IN const std::vector<cv::Mat> &imgs,
                       OUT PhaseSolverGroupDataHost &groupData,
                       IN const float sncThreshold,
                       IN const int shiftTime = 4) = 0;
#ifdef __WITH_CUDA__
    /**
     * @brief 解相
     *
     * @param imgs 图像(CV_8UC1)
     * @param groupData 求解结果
     * @param sncThreshold 信噪比阈值
     * @param stream 异步流
     * @param block 块大小
     * @param shiftTime 相移步数
     */
    virtual void solve(IN const std::vector<cv::Mat>& imgs,
                       OUT PhaseSolverGroupDataDevice& groupData,
                       IN const float sncThreshold,
                       IN const int shiftTime = 4,
                       IN cv::cuda::Stream& stream = cv::cuda::Stream::Null(),
                       IN const dim3 block = dim3(32, 8)) = 0;
#endif //!__WITH_CUDA__
};
} // namespace phaseSolver
} // namespace sl
#endif // !__PHASESOLVER_PHASESOLVER_H_

