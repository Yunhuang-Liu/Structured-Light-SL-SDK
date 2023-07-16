/**
 * @file rectifierGpu.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  极线矫正器(GPU版本)
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */

#ifndef __RECTIFIER_RECTIFIER_GPU_H_
#define __RECTIFIER_RECTIFIER_GPU_H_

#include <rectifier.h>

/** @brief 结构光库 */
namespace sl {
/** @brief 极线校正库 */
namespace rectifier {
/** @brief GPU极线校正器 */
class RectifierGpu : public Rectifier {
  public:
    RectifierGpu();
    /**
     * @brief           使用标定参数构造极线校正器
     *
     * @param info      相机标定参数
     */
    RectifierGpu(IN const tool::Info &info);
    /**
     * @brief           使用标定参数初始化极线校正器
     *
     * @param info      相机标定参数
     */
    void initialize(IN const tool::Info &info) override;
    /**
     * @brief                对图片进行极线校正
     *
     * @param imgInput        输入图片(CV_8UC1 || CV_32FC1)
     * @param imgOutput       输出图片(CV_8UC1 || CV_32FC1)
     * @param cvStream        异步流
     * @param isLeft          是否进行左相机校正：true 左，false 右
     */
    void
    remapImg(IN const cv::cuda::GpuMat &imgInput, OUT cv::cuda::GpuMat &imgOutput,
             IN const bool isLeft = true,
             IN cv::cuda::Stream &cvStream = cv::cuda::Stream::Null()) override;

  private:
    /**
     * @brief           对图片进行极线校正
     *
     * @param imgInput  输入图片(CV_8UC1 || CV_32FC1)
     * @param imgOutput 输出图片(CV_8UC1 || CV_32FC1)
     * @param isLeft    是否进行左相机校正：true 左，false 右
     */
    void remapImg(IN const cv::Mat &imgInput, OUT cv::Mat &imgOutput,
                  IN const bool isLeft = true) override {}
    //图像大小
    cv::Size __imgSize;
    //左相机X方向映射表
    cv::cuda::GpuMat __mapLx;
    //右相机Y方向映射表
    cv::cuda::GpuMat __mapLy;
    //左相机X方向映射表
    cv::cuda::GpuMat __mapRx;
    //右相机Y方向映射表
    cv::cuda::GpuMat __mapRy;
};
} // namespace rectifier
} // namespace sl
#endif // __RECTIFIER_RECTIFIER_GPU_H_