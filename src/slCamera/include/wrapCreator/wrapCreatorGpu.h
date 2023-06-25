/**
 * @file wrapCreatorGpu.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief  GPU加速包裹相位求解器
 * @version 0.1
 * @date 2022-08-25
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __WRAPCREATOR_WRAPCREATOR_GPU_H_
#define __WRAPCREATOR_WRAPCREATOR_GPU_H_

#include <wrapCreator.h>

/** @brief 结构光库 */
namespace sl {
/** @brief 包裹生成库 */
namespace wrapCreator {
/** @brief GPU加速包裹相位求解器 */
class WrapCreatorGpu : public WrapCreator {
  public:
    /**
     * @brief                   求取包裹相位
     *
     * @param imgs              输入，相移图片(CV_8UC1)
     * @param wrapImg           输入，包裹图片(CV_32FC1)
     * @param conditionImg      输入，背景图片(CV_32FC1)
     * @param cvStream          输入，非阻塞流
     * @param block             输入，加速参数
     */
    void getWrapImg(IN const std::vector<cv::Mat> &imgs,
                    OUT cv::cuda::GpuMat &wrapImg,
                    OUT cv::cuda::GpuMat &conditionImg,
                    IN cv::cuda::Stream &cvStream = cv::cuda::Stream::Null(),
                    IN const dim3 block = dim3(32, 8)) override;

  private:
    void getWrapImg(const std::vector<cv::Mat> &imgs, cv::Mat &wrapImg,
                    cv::Mat &conditionImg) override {}
};
} // namespace wrapCreator
} // namespace sl
#endif // !__WRAPCREATOR_WRAPCREATOR_GPU_H_