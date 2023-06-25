/**
 * @file wrapCreator.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief  包裹相位求解器
 * @version 0.1
 * @date 2022-08-25
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __WRAPCREATOR_WRAPERCREATOR_H_
#define __WRAPCREATOR_WRAPERCREATOR_H_

#include <opencv2/opencv.hpp>

#include <typeDef.h>

#ifdef __WITH_CUDA__
#include <cuda_runtime.h>
#endif //!__WITH_CUDA__

/** @brief 结构光库 */
namespace sl {
/** @brief 包裹生成库 */
namespace wrapCreator {
#ifdef __WITH_CUDA__
/** @brief cuda函数库 */
namespace cudaFunc {
void getWrapImgSync(IN const std::vector<cv::Mat> &imgs,
                    OUT cv::cuda::GpuMat &wrapImg,
                    OUT cv::cuda::GpuMat &conditionImg,
                    IN cv::cuda::Stream &cvStream = cv::cuda::Stream::Null(),
                    IN const dim3 block = dim3(32, 8));
}
#endif //!__WITH_CUDA__

/** @brief 包裹求解器 */
class WrapCreator {
  public:
    virtual ~WrapCreator() {}
    /**
     * @brief 获取包裹相位
     *
     * @param imgs            输入，图片(CV_8UC1 || CV_32FC1)
     * @param wrapImg         输出，包裹图片(CV_32FC1)
     * @param conditionImg    输出，调制图片(CV_32FC1)
     */
    virtual void getWrapImg(IN const std::vector<cv::Mat> &imgs,
                            OUT cv::Mat &wrapImg,
                            OUT cv::Mat &conditionImg) = 0;
#ifdef __WITH_CUDA__
    /**
     * @brief 获取包裹相位
     *
     * @param imgs            输入，图片(CV_8UC1)
     * @param wrapImg         输出，包裹图片(CV_32FC1)
     * @param conditionImg    输出，调制图片(CV_32FC1)
     * @param cvStream        输入，异步流
     * @param parameter       输入，算法加速控制参数
     */
    virtual void
    getWrapImg(IN const std::vector<cv::Mat> &imgs,
               OUT cv::cuda::GpuMat &wrapImg,
               OUT cv::cuda::GpuMat &conditionImg,
               IN cv::cuda::Stream &cvStream = cv::cuda::Stream::Null(),
               IN const dim3 block = dim3(32, 8)) = 0;
#endif //!__WITH_CUDA__
};
} // namespace wrapCreator
} // namespace sl
#endif // !__WRAPCREATOR_WRAPERCREATOR_H_