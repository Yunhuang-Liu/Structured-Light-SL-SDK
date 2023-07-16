#include <restructorGpu.h>

#include <limits>

namespace sl {
namespace restructor {
RestructorGpu::RestructorGpu(const tool::Info &calibrationInfo)
    : __calibrationInfo(calibrationInfo) {
    cv::Mat QCv, R1InvCv, M1Cv, M3Cv, RCv, TCv;
    __calibrationInfo.__M1.convertTo(M1Cv, CV_32FC1);
    __calibrationInfo.__Q.convertTo(QCv, CV_32FC1);
    __calibrationInfo.__R1.convertTo(R1InvCv, CV_32FC1);
    R1InvCv = R1InvCv.inv();
    cv::cv2eigen(QCv, __Q);
    cv::cv2eigen(R1InvCv, __R1Inv);
    cv::cv2eigen(M1Cv, __M1);
    if (!__calibrationInfo.__M3.empty()) {
        __calibrationInfo.__M3.convertTo(M3Cv, CV_32FC1);
        __calibrationInfo.__Rlr.convertTo(RCv, CV_32FC1);
        __calibrationInfo.__Tlr.convertTo(TCv, CV_32FC1);
        cv::cv2eigen(M3Cv, __M3);
        cv::cv2eigen(RCv, __R);
        cv::cv2eigen(TCv, __T);
    }
}

void RestructorGpu::restruction(const cv::cuda::GpuMat &leftAbsImg,
                                const cv::cuda::GpuMat &rightAbsImg,
                                const RestructParamater param,
                                cv::cuda::GpuMat &depthImg,
                                cv::cuda::Stream &stream) {
    if (depthImg.empty()) {
        depthImg.create(leftAbsImg.rows, leftAbsImg.cols, CV_32FC1);
    } else {
        depthImg.setTo(0);
    }

    getDepthMap(leftAbsImg, rightAbsImg, param, depthImg, stream);
}

void RestructorGpu::getDepthMap(const cv::cuda::GpuMat &leftImg,
                                const cv::cuda::GpuMat &rightImg,
                                const RestructParamater param,
                                cv::cuda::GpuMat &depthImg,
                                cv::cuda::Stream &pStream) {
    restructor::cudaFunc::getDepthMap(leftImg, rightImg, param, __Q, __M1, __M3,
                                      __R, __T, __R1Inv, depthImg, pStream, param.__block);
}
} // namespace restructor
} // namespace sl
