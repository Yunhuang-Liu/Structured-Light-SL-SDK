#include <nStepNGrayCodeMasterGpu.h>

namespace sl {
namespace phaseSolver {
void NStepNGrayCodeMasterGpu::solve(IN const std::vector<cv::Mat> &imgs,
                                    OUT PhaseSolverGroupDataDevice &groupData,
                                    IN const float sncThreshold,
                                    IN const int shiftTime,
                                    IN cv::cuda::Stream &stream,
                                    IN const dim3 block) {
    const int rows = imgs[0].rows;
    const int cols = imgs[0].cols;

    groupData.__unwrapMap.create(rows, cols, CV_32FC1);
    groupData.__textureMap.create(rows, cols, CV_32FC1);
    groupData.__wrapMap.create(rows, cols, CV_32FC1);

    sl::phaseSolver::cudaFunc::solvePhase(imgs, groupData, sncThreshold,
                                          shiftTime, stream, block);
}
} // namespace phaseSolver
} // namespace sl