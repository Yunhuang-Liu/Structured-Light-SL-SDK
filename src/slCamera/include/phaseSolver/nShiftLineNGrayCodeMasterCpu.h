/**
 * @file nShiftLineNGrayCodeMasterCpu.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  任意步线移任意位格雷码解相器
 * @version 0.1
 * @date 2022-10-06
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __PHASESOLVER_NSHIFTLINENGRAYCODEMASTER_CPU_H_
#define __PHASESOLVER_NSHIFTLINENGRAYCODEMASTER_CPU_H_

#include <phaseSolver.h>

#include <fstream>
#include <vector>

/** @brief 结构光库 */
namespace sl {
/** @brief 解相库 */
namespace phaseSolver {
/**
 * @brief N位格雷码N步线移解码器(多线程+SIMD)
 */
class NShiftLineNGrayCodeMasterCpu : public PhaseSolver {
  public:
    /**
     * @brief 解算
     * @note
     * imgs组成为grayCodeBits张格雷码图片，剩余图片前一半为正线移图案，后一半为反线移图案
     *
     * @param imgs 图像(CV_8UC1 || CV_32FC1)
     * @param groupData 解算所得数据
     * @param sncThreshold 信噪比阈值
     * @param grayCodeBits 格雷码位数
     */
    void solve(IN const std::vector<cv::Mat> &imgs,
               OUT PhaseSolverGroupDataHost &groupData,
               IN const float sncThreshold,
               IN const int grayCodeBits = 9) override;

  private:
#ifdef __WITH_CUDA__
    /**
     * @brief 解相
     * @note
     * imgs组成为grayCodeBits张格雷码图片，剩余图片前一半为正线移图案，后一半为反线移图案
     *
     * @param imgs 图像(CV_8UC1)
     * @param groupData 解算数据
     * @param sncThreshold 信噪比阈值
     * @param stream 异步流
     * @param block 块大小
     * @param grayCodeBits 相移步数
     */
    void solve(IN const std::vector<cv::Mat> &imgs,
               OUT PhaseSolverGroupDataDevice &groupData,
               IN const float sncThreshold, IN const int grayCodeBits = 4,
               IN cv::cuda::Stream &stream = cv::cuda::Stream::Null(),
               IN const dim3 block = dim3(32, 8)) override {}
#endif //!__WITH_CUDA__
    /**
     * @brief 通过正反位移图案定位亚像素边缘
     *
     * @param positiveImg 正位移图案
     * @param negtiveImg 反位移图案
     * @param allocateEdgeCode 位移码值
     * @param shiftLineCodeImg 位移码值图
     * @param edgeImg 边缘码值图
     */
    void locateSubpixelEdge(IN const cv::Mat &positiveImg,
                            IN const cv::Mat &negtiveImg,
                            IN const int allocateEdgeCode,
                            cv::Mat &shiftLineCodeImg, cv::Mat &edgeImg);
    /**
     * @brief 格雷码解码
     *
     * @param imgs             图片(CV_32FC1)
     * @param grayCodeBits     格雷码位数
     * @param conditionImg     阈值图像(CV_32FC1)
     * @param shiftCodeImg  移动码值图像(CV_32FC1)
     * @param grayCodeImg      码值图片(CV_32FC1)
     */
    void decodeGrayCode(IN const std::vector<cv::Mat> &imgs,
                        IN const int grayCodeBits,
                        IN const cv::Mat &conditionImg,
                        IN const cv::Mat &shiftCodeImg,
                        OUT cv::Mat &grayCodeImg);
    /**
     * @brief 0位线移特征码值解码
     *
     * @param imgs          图片(CV_32FC1)
     * @param grayCodeBits  格雷码码数
     * @param conditionImg  阈值图像(CV_32FC1)
     * @param shiftCodeImg  移动码值图像(CV_32FC1)
     * @param maskCodeImg   码值图像(CV_32FC1)
     */
    void decodeShiftLineMaskCode(IN const std::vector<cv::Mat> &imgs,
                                 IN const int grayCodeBits,
                                 IN const cv::Mat &conditionImg,
                                 IN const cv::Mat &shiftCodeImg,
                                 OUT cv::Mat &maskCodeImg);
    /**
     * @brief 线程入口函数，0位线移特征码值解码
     *
     * @param imgs          图片(CV_32FC1)
     * @param grayCodeBits  格雷码码数
     * @param conditionImg  阈值图像(CV_32FC1)
     * @param shiftCodeImg  移动码值图像(CV_32FC1)
     * @param region        图像操作范围
     * @param maskCodeImg   码值图像(CV_32FC1)
     */
    void entryDecodeShiftLineMaskCode(IN const std::vector<cv::Mat> &imgs,
                                      IN const int grayCodeBits,
                                      IN const cv::Mat &conditionImg,
                                      IN const cv::Mat& shiftCodeImg,
                                      IN const cv::Point2i region,
                                      OUT cv::Mat &maskCodeImg);
    /**
     * @brief 线程入口函数，多线程解码格雷码
     *
     * @param imgs             图片(CV_32FC1)
     * @param grayCodeBits     格雷码位数
     * @param conditionImg     阈值图像(CV_32FC1)
     * @param shiftCodeImg     移动码值图像(CV_32FC1)
     * @param region           图像操作范围
     * @param grayCodeImg      码值图片(CV_32FC1)
     */
    void entryDecodeGrayCode(IN const std::vector<cv::Mat> &imgs,
                             IN const int grayCodeBits,
                             IN const cv::Mat &conditionImg,
                             IN const cv::Mat &shiftCodeImg,
                             IN const cv::Point2i region,
                             OUT cv::Mat &grayCodeImg);
    /**
     * @brief 亚像素线移图案边缘多线程定位入口函数
     *
     * @param diffImg 差异图像(CV_32C1)
     * @param positiveImg 正线移图案(CV_32C1)
     * @param negtiveImg 负线移图案(CV_32FC1)
     * @param allocateEdgeCode 输入，分配的线移码值
     * @param region 图像操作范围
     * @param shiftLineCodeImg 线移图案线移码值图(CV_32FC1)
     * @param edgeImg 线移图案呀亚像素边缘码值图(CV_32FC1)
     */
    void entryLocateSubpixelEdge(IN const cv::Mat &diffImg,
                                 IN const cv::Mat &positiveImg,
                                 IN const cv::Mat &negtiveImg,
                                 IN const int allocateEdgeCode,
                                 IN const cv::Point2i region,
                                 OUT cv::Mat &shiftLineCodeImg,
                                 OUT cv::Mat &edgeImg);
    /**
     * @brief 计算两线交点
     *
     * @param lineA 线段A
     * @param lineB 线段B
     * @return cv::Point2f 交点
     */
    cv::Point2f getCrossPoint(IN cv::Vec4f lineA, IN cv::Vec4f lineB);
};
} // namespace phaseSolver
} // namespace sl
#endif //__PHASESOLVER_NSHIFTLINENGRAYCODEMASTER_CPU_H_
