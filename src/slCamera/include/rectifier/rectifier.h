/**
 * @file rectifier.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  极线校正器
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */

#ifndef __RECTIFIER_RECTIFIER_H_
#define __RECTIFIER_RECTIFIER_H_

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

#include <matrixsInfo.h>
#include <typeDef.h>

/** @brief 结构光库 */
namespace sl {
/** @brief 极线校正库 */
namespace rectifier {
/** @brief 极线校正器 */
class Rectifier {
  public:
    virtual ~Rectifier() {}
    /**
     * @brief           使用标定参数初始化极线校正器
     *
     * @param info      相机标定参数
     */
    virtual void initialize(IN const tool::Info &info) = 0;
    /**
     * @brief           对图片进行极线校正
     *
     * @param imgInput  输入图片(CV_8UC1 || CV_32FC1)
     * @param imgOutput 输出图片(CV_8UC1 || CV_32FC1)
     * @param isLeft    是否进行左相机校正：true 左，false 右
     */
    virtual void remapImg(IN const cv::Mat &imgInput, OUT cv::Mat &imgOutput,
                          IN const bool isLeft = true) = 0;
#ifdef __WITH_CUDA__
    /**
     * @brief           对图片进行极线校正
     *
     * @param imgInput  输入图片(CV_8UC1 || CV_32FC1)
     * @param imgOutput 输出图片(CV_8UC1 || CV_32FC1)
     * @param isLeft    是否进行左相机校正：true 左，false 右
     * @param cvStream  异步流
     */
    virtual void
    remapImg(IN const cv::cuda::GpuMat& imgInput, OUT cv::cuda::GpuMat &imgOutput,
             IN const bool isLeft = true,
             IN cv::cuda::Stream &cvStream = cv::cuda::Stream::Null()) = 0;
#endif // !__WITH_CUDA__
  private:
};
} // namespace rectifier
} // namespace sl
#endif // __RECTIFIER_RECTIFIER_H_
