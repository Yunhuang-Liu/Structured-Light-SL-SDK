/**
 * @file restructor_GPU.h
 * @author Liu Yunhuang(1369215984@qq.com)
 * @brief  GPU重建器
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef __RESTRUCTOR_RESTRUCTOR_GPU_H_
#define __RESTRUCTOR_RESTRUCTOR_GPU_H_

#include <restructor.h>
#include <matrixsInfo.h>

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

#include <device_launch_parameters.h>
#include <cuda_runtime.h>

/** @brief 结构光库 */
namespace sl {
    /** @brief 重建库 */
    namespace restructor {
        /** @brief GPU加速重建器 */
        class RestructorGpu : public Restructor {
        public:
            /**
             * @brief 构造函数
             * 
             * @param calibrationInfo 标定信息
             */
            RestructorGpu(IN const tool::Info &calibrationInfo);
            /**
             * @brief 重建
             * 
             * @param leftAbsImg 左绝对相位(CV_32FC1)
             * @param rightAbsImg 右绝对相位(CV_32FC1)
             * @param param 重建器控制参数
             * @param depthImg 深度图(CV_32FC1)
             * @param stream 输入，异步流
             */
            void restruction(IN const cv::cuda::GpuMat &leftAbsImg,
                             IN const cv::cuda::GpuMat &rightAbsImg,
                             IN const RestructParamater param,
                             OUT cv::cuda::GpuMat &depthImg,
                             IN cv::cuda::Stream &stream = cv::cuda::Stream::Null()) override;
        protected:
            /**
             * @brief 映射深度纹理
             * 
             * @param leftImg 输入，左绝对相位(CV_32FC1)
             * @param rightImg 输入，右绝对相位(CV_32FC1)
             * @param param 重建器控制参数
             * @param depthImg 输入/输出，深度图(CV_32FC1)
             * @param pStream 输入，异步流
             */
            void getDepthMap(IN const cv::cuda::GpuMat &leftImg,
                             IN const cv::cuda::GpuMat &rightImg,
                             IN const RestructParamater param,
                             OUT cv::cuda::GpuMat &depthImg,
                             IN cv::cuda::Stream &pStream = cv::cuda::Stream::Null());

        private:
            /** \标定信息 **/
            const tool::Info &__calibrationInfo;
            //CPU端函数
            void restruction(IN const cv::Mat &leftAbsImg,
                     IN const cv::Mat &rightAbsImg,
                     IN const RestructParamater param,
                     OUT cv::Mat &depthImgOut)  override {}
            /** \深度映射矩阵 **/
            Eigen::Matrix4f __Q;
            /** \深度映射矩阵 **/
            Eigen::Matrix3f __R1Inv;
            /** \灰度相机到彩色相机旋转矩阵 **/
            Eigen::Matrix3f __R;
            /** \灰度相机到彩色相机位移矩阵 **/
            Eigen::Vector3f __T;
            /** \点云相机内参矩阵 **/
            Eigen::Matrix3f __M1;
            /** \彩色相机内参矩阵 **/
            Eigen::Matrix3f __M3;
        };
    }// namespace restructor
}// namespace sl
#endif // !__RESTRUCTOR_RESTRUCTOR_GPU_H_
