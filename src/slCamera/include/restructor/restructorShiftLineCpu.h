/**
 * @file restructorShiftLineCpu.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  CPU重建器
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef __RESTRUCTOR_RESTRUCTOR_SHIFTLINE_CPU_H_
#define __RESTRUCTOR_RESTRUCTOR_SHIFTLINE_CPU_H_

#include <matrixsInfo.h>
#include <restructor.h>


#include <immintrin.h>
#include <limits>

/** @brief 结构光库 */
namespace sl {
/** @brief 重建库 */
namespace restructor {
/** @brief CPU加速重建器 */
class RestructorShiftLineCpu : public Restructor {
  public:
    /**
     * @brief 构造函数
     *
     * @param calibrationInfo 标定信息
     */
    RestructorShiftLineCpu(IN const tool::Info &calibrationInfo);
    /**
     * @brief 重建
     *
     * @param leftCodeImg 左码值图(CV_32FC4)
     * @param rightCodeImg 右码值图(CV_32FC4)
     * @param param     重建器控制参数
     * @param depthImgOut 深度图(CV_32FC1)
     */
    void restruction(IN const cv::Mat &leftAbsImg,
                     IN const cv::Mat &rightAbsImg,
                     IN const RestructParamater param,
                     OUT cv::Mat &depthImgOut) override;

  protected:
    /**
     * @brief 获取深度纹理
     *
     * @param leftCodeImg 左码值图(CV_32FC4)
     * @param rightCodeImg 右码值图(CV_32FC4)
     * @param param 重建器控制参数
     * @param depthImgOut 深度图(CV_32FC1)
     */
    void getDepthColorMap(IN const cv::Mat &leftCodeImg,
                          IN const cv::Mat &rightCodeImg,
                          IN const RestructParamater param,
                          OUT cv::Mat &depthImgOut);

  private:
#ifdef __WITH_CUDA__
    void restruction(
        IN const cv::cuda::GpuMat &leftAbsImg,
        IN const cv::cuda::GpuMat &rightAbsImg,
        IN const RestructParamater param, OUT cv::cuda::GpuMat &depthImg,
        IN cv::cuda::Stream &stream = cv::cuda::Stream::Null()) override {}
#endif //!__WITH_CUDA__
    /** \存储线程锁 **/
    std::mutex __mutexMap;
    /** \标定信息 **/
    const tool::Info &__calibrationInfo;
    /**
     * @brief 获取深度纹理入口函数
     *
     * @param leftCodeImg 左码值图(CV_32FC1)
     * @param rightCodeImg 右码值图(CV_32FC1)
     * @param param 重建器重建参数
     * @param depthImgOut 深度图(CV_32FC1)
     * @param region 执行区域（行）
     */
    void entryDepthColorMap(const cv::Mat &leftCodeImg,
                            const cv::Mat &rightCodeImg,
                            const RestructParamater param, cv::Mat &depthImgOut,
                            const cv::Point2i region);
};
} // namespace restructor
} // namespace sl
#endif // !__RESTRUCTOR_RESTRUCTOR_SHIFTLINE_CPU_H_
