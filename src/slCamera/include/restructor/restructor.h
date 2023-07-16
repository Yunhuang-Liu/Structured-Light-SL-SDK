/**
 * @file restructor.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  重建器
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef __RESTRUCTOR_RESTRUCTOR_H_
#define __RESTRUCTOR_RESTRUCTOR_H_

#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include <typeDef.h>

#ifdef __WITH_CUDA__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif // !__WITH_CUDA__

/** @brief 结构光库 */
namespace sl {
/** @brief 重建库 */
namespace restructor {
/** @brief 重建器控制参数 **/
struct RestructParamater {
    RestructParamater() {}
    RestructParamater(IN const int minDisparity, IN const int maxDisparity,
                      IN const float minDepth, IN const float maxDepth)
        : __minDisparity(minDisparity), __maxDisparity(maxDisparity),
          __minDepth(minDepth), __maxDepth(maxDepth),
          __isMapToPreDepthAxes(false), __isMapToColorCamera(false),
          __maximumCost(0.1) {
#ifdef __WITH_CUDA__
        __block = dim3(32, 8);
#endif //!__WITH_CUDA__
    }
    RestructParamater(IN const int minDepth, IN const int maxDepth)
        : __minDisparity(-1000), __maxDisparity(1000), __minDepth(minDepth),
          __maximumCost(0.1), __maxDepth(maxDepth),
          __isMapToPreDepthAxes(false), __isMapToColorCamera(false) {
#ifdef __WITH_CUDA__
        __block = dim3(32, 8);
#endif //!__WITH_CUDA__
    }
    /** \最小视差 **/
    int __minDisparity;
    /** \最大视差 **/
    int __maxDisparity;
    /** \最小深度 **/
    float __minDepth;
    /** \最大深度 **/
    float __maxDepth;
    /** \匹配点最大代价 */
    float __maximumCost;
    /** \是否映射深度图至左相机坐标系 */
    bool __isMapToPreDepthAxes;
    /** \是否映射纹理 */
    bool __isMapToColorCamera;
#ifdef __WITH_CUDA__
    dim3 __block;
#endif // !__WITH_CUDA__
};

#ifdef __WITH_CUDA__
/** @brief cuda函数库 */
namespace cudaFunc {
//深度映射，CUDA主机端调用函数，不进行纹理映射
void getDepthMap(IN const cv::cuda::GpuMat &leftImg,
                 IN const cv::cuda::GpuMat &rightImg,
                 IN const RestructParamater param, const Eigen::Matrix4f &Q,
                 IN const Eigen::Matrix3f &M1, const Eigen::Matrix3f &M3,
                 IN const Eigen::Matrix3f &R, const Eigen::Vector3f &T,
                 IN const Eigen::Matrix3f &R1Inv,
                 OUT cv::cuda::GpuMat &depthMap,
                 IN cv::cuda::Stream &cvStream = cv::cuda::Stream::Null(),
                 IN const dim3 block = dim3(32, 8));
} // namespace cudaFunc
#endif // ! __WITH_CUDA__

/** @brief 重建器 */
class Restructor {
  public:
    virtual ~Restructor() {}
    /**
     * @brief 重建
     *
     * @param leftAbsImg  左绝对相位(CV_32FC1)
     * @param rightAbsImg 右绝对相位(CV_32FC1)
     * @param param       重建器控制参数
     * @param depthImgOut 深度图(CV_32FC1)
     */
    virtual void restruction(IN const cv::Mat &leftAbsImg,
                             IN const cv::Mat &rightAbsImg,
                             IN const RestructParamater param,
                             OUT cv::Mat &depthImgOut) = 0;
#ifdef __WITH_CUDA__
    /**
     * @brief 重建
     *
     * @param leftAbsImg  左绝对相位(CV_32FC1)
     * @param rightAbsImg 右绝对相位(CV_32FC1)
     * @param param       重建器控制参数
     * @param depthImg      输出深度图(CV_32FC1)
     * @param stream      异步流
     */
    virtual void
    restruction(IN const cv::cuda::GpuMat &leftAbsImg,
                IN const cv::cuda::GpuMat &rightAbsImg,
                IN const RestructParamater param,
                OUT cv::cuda::GpuMat &depthImg,
                IN cv::cuda::Stream &stream = cv::cuda::Stream::Null()) = 0;
#endif // !__WITH_CUDA__
  protected:
    /**
     * @brief 获取深度和纹理
     */
    void getDepthMap() {}
};
} // namespace restructor
} // namespace sl
#endif // __RESTRUCTOR_RESTRUCTOR_H_
