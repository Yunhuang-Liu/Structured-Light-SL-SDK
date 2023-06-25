/**
 * @file wrapCreatorCpu.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief  CPU加速包裹相位求解器
 * @version 0.1
 * @date 2022-08-25
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __WRAPCREATOR_WRAPCREATOR_CPU_H_
#define __WRAPCREATOR_WRAPCREATOR_CPU_H_

#include <wrapCreator.h>

#include <immintrin.h>

#include <thread>

/** @brief 结构光库 */
namespace sl {
/** @brief 包裹生成库 */
namespace wrapCreator {
/** @brief CPU加速包裹相位求解器 */
class WrapCreatorCpu : public WrapCreator {
  public:
    /**
     * @brief                   获取包裹相位
     *
     * @param imgs              输入，相移图片(CV_8UC1 || CV_32FC1)
     * @param wrapImg           输出，包裹图片(CV_32FC1)
     * @param conditionImg      输出，背景图片(CV_32FC1)
     */
    void getWrapImg(IN const std::vector<cv::Mat> &imgs, OUT cv::Mat &wrapImg,
                    OUT cv::Mat &conditionImg) override;

  protected:
#ifdef __WITH_CUDA__
    void getWrapImg(IN const std::vector<cv::Mat> &imgs,
                    OUT cv::cuda::GpuMat &wrapImg,
                    OUT cv::cuda::GpuMat &conditionImg,
                    IN cv::cuda::Stream &cvStream = cv::cuda::Stream::Null(),
                    IN const dim3 block = dim3(32, 8)) override{};
#endif //!__WITH_CUDA__
  private:
    /**
     * @brief 求解调制图像
     * @note
     * 由于条纹生成时将环境光强与表面调制光强设为1:1，因此调制图像与纹理图像应当相等
     *
     * @param imgs 相移图像(CV_32FC1)
     * @param conditionImg 调制图像(CV_32FC1)
     */
    void solveTextureImgs(IN const std::vector<cv::Mat> &imgs,
                          OUT cv::Mat &conditionImg);
    /**
     * @brief 线程入口函数，多线程求解调制图像
     *
     * @param imgs          输入，图片(CV_32FC1)
     * @param region        输入，图像操作区间
     * @param conditionImg  输出，调制图像(CV_32FC1)
     */
    void entrySolveTextureImgs(IN const std::vector<cv::Mat> &imgs,
                               IN const cv::Point2i region,
                               OUT cv::Mat &conditionImg);
    /**
     * @brief 求解包裹图像入口函数
     *
     * @param imgs 格雷相移图像(CV_32FC1)
     * @param region 区域（行）
     * @param wrapImg 包裹图像(CV_32FC1)
     */
    void entryWrap(IN const std::vector<cv::Mat> &imgs,
                   IN const cv::Point2i &region, OUT cv::Mat &wrapImg);
};
} // namespace wrapCreator
} // namespace sl
#endif // __WRAPCREATOR_WRAPCREATOR_CPU_H_