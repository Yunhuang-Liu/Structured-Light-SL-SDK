/**
 * @file nStepNGrayCodeMasterGpu.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  任意步任意位互补格雷码解相器
 * @version 0.1
 * @date 2022-10-06
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __PHASESOLVER_NSTEPNGRAYCODEMASTER_GPU_H_
#define __PHASESOLVER_NSTEPNGRAYCODEMASTER_GPU_H_

#include <phaseSolver.h>

#include <fstream>
#include <vector>

/** @brief 结构光库 */
namespace sl {
/** @brief 解相库 */
namespace phaseSolver {
/**
 * @brief N位互补格雷码N步相移解码器(CUDA)
 */
class NStepNGrayCodeMasterGpu : public PhaseSolver {
  public:
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
               IN const dim3 block = dim3(32, 8)) override;

  private:
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
               IN const int shiftTime = 4) override {};
};
} // namespace phaseSolver
} // namespace sl
#endif //__PHASESOLVER_NSTEPNGRAYCODEMASTER_CPU_H_
