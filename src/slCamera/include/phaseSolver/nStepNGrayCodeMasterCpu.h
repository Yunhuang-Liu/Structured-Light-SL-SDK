/**
 * @file nStepNGrayCodeMasterCpu.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  任意步任意位互补格雷码解相器
 * @version 0.1
 * @date 2022-10-06
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __PHASESOLVER_NSTEPNGRAYCODEMASTER_CPU_H_
#define __PHASESOLVER_NSTEPNGRAYCODEMASTER_CPU_H_

#include <phaseSolver.h>

#include <fstream>
#include <vector>

/** @brief 结构光库 */
namespace sl {
/** @brief 解相库 */
namespace phaseSolver {
/**
 * @brief N位互补格雷码N步相移解码器(多线程+SIMD)
 */
class NStepNGrayCodeMasterCpu : public PhaseSolver {
  public:
    /**
     * @brief 解算
     *
     * @param imgs 图像(CV_8UC1 || CV_32FC1)
     * @param groupData 解算所得数据
     * @param sncThreshold 信噪比阈值
     * @param shiftTime 相移次数
     */
    void solve(IN const std::vector<cv::Mat> &imgs,
               OUT PhaseSolverGroupDataHost &groupData,
               IN const float sncThreshold,
               IN const int shiftTime = 4) override;

  private:
#ifdef __WITH_CUDA__
    /**
     * @brief 解相
     *
     * @param imgs 图像(CV_8UC1)
     * @param groupData 解算数据
     * @param sncThreshold 信噪比阈值
     * @param stream 异步流
     * @param block 块大小
     * @param shiftTime 相移步数
     */
    void solve(IN const std::vector<cv::Mat> &imgs,
               OUT PhaseSolverGroupDataDevice &groupData,
               IN const float sncThreshold, 
               IN const int shiftTime = 4,
               IN cv::cuda::Stream &stream = cv::cuda::Stream::Null(),
               IN const dim3 block = dim3(32, 8)) override {}
#endif //!__WITH_CUDA__
    /**
     * @brief 计算包裹图像
     *
     * @param imgs 图像(CV_32FC1)
     * @param wrapImg 包裹图像(CV_32FC1)
     * @param conditionImg 调制图像(CV_32FC1)
     * @param shiftStep 位移步数
     */
    void getPrepareDate(IN const std::vector<cv::Mat> &imgs,
                        OUT cv::Mat &wrapImg, cv::Mat &conditionImg,
                        IN const int shiftStep = 4);
    /**
     * @brief 求解调制图像
     * @note
     * 由于条纹生成时将环境光强与表面调制光强设为1:1，因此调制图像与纹理图像应当相等
     *
     * @param imgs 相移格雷码图像(CV_32FC1)
     * @param conditionImg 调制图像(CV_32FC1)
     * @param shiftStep 相移步数
     */
    void solveTextureImgs(IN const std::vector<cv::Mat> &imgs,
                          OUT cv::Mat &conditionImg,
                          IN const int shiftStep = 4);
    /**
     * @brief 线程入口函数，多线程求解调制图像
     *
     * @param imgs          输入，图片(CV_32FC1)
     * @param shiftStep     输入，相移步数
     * @param region        输入，图像操作区间
     * @param conditionImg  输出，调制图像(CV_32FC1)
     */
    void entrySolveTextureImgs(IN const std::vector<cv::Mat> &imgs,
                               IN const int shiftStep,
                               IN const cv::Point2i region,
                               OUT cv::Mat &conditionImg);
    /**
     * @brief 线程入口函数，多线程解相
     *
     * @param imgs             输入，图片(CV_32FC1)
     * @param shiftStep        输入，相移步数
     * @param conditionImg     输入，阈值图像(CV_32FC1)
     * @param sncThreshold     输入，信噪比阈值(CV_32FC1)
     * @param wrapImg          输入，包裹图像(CV_32FC1)
     * @param region           输入，图像操作范围
     * @param absolutePhaseImg 输出，绝对相位图片(CV_32FC1)
     */
    void entryUnwrap(IN const std::vector<cv::Mat> &imgs,
                     IN const int shiftStep, IN const cv::Mat &conditionImg,
                     IN const float sncThreshold, IN const cv::Mat &wrapImg,
                     IN const cv::Point2i region,
                     OUT cv::Mat &absolutePhaseImg);
    /**
     * @brief 求解包裹图像入口函数
     *
     * @param imgs 格雷相移图像(CV_32FC1)
     * @param shiftStep 相移步数
     * @param region 区域（行）
     * @param wrapImg 包裹图像(CV_32FC1)
     */
    void entryWrap(IN const std::vector<cv::Mat> &imgs, IN const int shiftStep,
                   IN const cv::Point2i &region, OUT cv::Mat &wrapImg);
};
} // namespace phaseSolver
} // namespace sl
#endif //__PHASESOLVER_NSTEPNGRAYCODEMASTER_CPU_H_
