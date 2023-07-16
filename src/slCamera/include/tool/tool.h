/**
 * @file tool.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief  工具库
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef __TOOL_TOOL_H_
#define __TOOL_TOOL_H_

#ifdef __WITH_CUDA__
//#include <cuda/std/cmath>
#include <cuda/std/functional>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudalegacy.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#endif //!__WITH_CUDA__

#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <matrixsInfo.h>
#include <typeDef.h>

/** @brief 结构光库 **/
namespace sl {
/** @brief 工具库 **/
namespace tool {
#ifdef __WITH_CUDA__
/** @brief cuda函数库 */
namespace cudaFunc {
/**
 * @brief          由相移图片计算纹理图片(灰度图片不支持，请直接使用包裹求解器或者相位求解器获得纹理)
 *
 * @param imgs     相移图片(CV_8UC3)
 * @param texture  纹理图片(CV_8UC3)
 * @param block    线程块
 * @param stream   异步流
 */
void averageTexture(IN const std::vector<cv::Mat> &imgs,
                    OUT cv::cuda::GpuMat &texture,
                    IN const dim3 block = dim3(32, 8),
                    IN cv::cuda::Stream &cvStream = cv::cuda::Stream::Null());
/**
 * @brief               全图像相位高度映射（CUDA加速优化）
 *
 * @param phase         相位图(CV_32FC1)
 * @param intrinsic     内参
 * @param coefficient   八参数
 * @param minDepth      最小深度
 * @param maxDepth      最大深度
 * @param depth         深度图(CV_32FC1)
 * @param block         线程块
 * @param stream        异步流
 */
void phaseHeightMapEigCoe(
    IN const cv::cuda::GpuMat &phase, IN const Eigen::Matrix3f &intrinsic,
    IN const Eigen::Vector<float, 8> &coefficient, IN const float minDepth,
    IN const float maxDepth, OUT cv::cuda::GpuMat &depth,
    IN const dim3 block = dim3(32, 8),
    IN cv::cuda::Stream &cvStream = cv::cuda::Stream::Null());
/**
 * @brief                     反向投影映射纹理（CUDA加速优化）
 *
 * @param depth               相位图(CV_32FC1)
 * @param textureSrc          深度图(CV_32FC1)
 * @param intrinsicInvD       第一个辅助相机的包裹相位(CV_32FC1)
 * @param intrinsicT          第一个辅助相机的调制图像(CV_32FC1)
 * @param rotateDToT          第二个辅助相机的包裹相位(CV_32FC1)
 * @param translateDtoT       第二个辅助相机的调制图像(CV_32FC1)
 * @param textureMapped       深度相机的内参逆矩阵
 * @param block               线程块
 * @param stream              异步流
 */
void reverseMappingTexture(
    IN const cv::cuda::GpuMat &depth, IN const cv::cuda::GpuMat &textureSrc,
    IN const Eigen::Matrix3f &intrinsicInvD,
    IN const Eigen::Matrix3f &intrinsicT, IN const Eigen::Matrix3f &rotateDToT,
    IN const Eigen::Vector3f &translateDtoT,
    OUT cv::cuda::GpuMat &textureMapped, IN const dim3 block = dim3(32, 8),
    IN cv::cuda::Stream &cvStream = cv::cuda::Stream::Null());
/**
 * @brief                     过滤相位
 *
 * @param absPhase            需要过滤的绝对相位
 * @param out                 过滤后的绝对相位
 * @param maxTollerance       允许最大不同量
 * @param kenel               卷积核大小
 * @param block               线程块
 * @param stream              异步流
 */
void filterPhase(IN const cv::cuda::GpuMat &absPhase, OUT cv::cuda::GpuMat &out,
                 IN const float maxTollerance, IN const int kernel,
                 IN const dim3 block = dim3(32, 8),
                 IN cv::cuda::Stream &cvStream = cv::cuda::Stream::Null());
} // namespace cudaFunc
#endif //!__WITH_CUDA__

/**
 * @brief 全图像相位高度映射
 *
 * @param phase         相位图(CV_32FC1)
 * @param intrinsic     内参(CV_64FC1)
 * @param coefficient   八参数(CV_64FC1)
 * @param minDepth      最小深度
 * @param maxDepth      最大深度
 * @param depth         深度图(CV_32FC1)
 * @param threads       使用的线程数
 */
void phaseHeightMapEigCoe(IN const cv::Mat &phase, IN const cv::Mat &intrinsic,
                          IN const cv::Mat &coefficient,
                          IN const float minDepth, IN const float maxDepth,
                          OUT cv::Mat &depth);
/**
 * @brief 区域图像相位高度映射
 *
 * @param phase         相位图(CV_32FC1)
 * @param intrinsic     内参(CV_64FC1)
 * @param coefficient   八参数(CV_64FC1)
 * @param minDepth      最小深度
 * @param maxDepth      最大深度
 * @param rowBegin      区域行起始位置
 * @param rowEnd        区域行结束位置
 * @param depth         深度图(CV_32FC1)
 */
void phaseHeightMapEigCoeRegion(IN const cv::Mat &phase,
                                IN const cv::Mat &intrinsic,
                                IN const cv::Mat &coefficient,
                                IN const float minDepth,
                                IN const float maxDepth, IN const int rowBegin,
                                IN const int rowEnd, OUT cv::Mat &depth);
/**
 * @brief                由相移图片计算纹理图片
 *
 * @param imgs           相移图片(CV_8UC3 || CV_8UC1)
 * @param texture        纹理图片(CV_8UC3 || CV_8UC1)
 * @param phaseShiftStep 相移步数（@note 前N张图片为相移图片）
 */
void averageTexture(IN const std::vector<cv::Mat> &imgs, OUT cv::Mat &texture,
                    IN const int phaseShiftStep = 4);
/**
 * @brief                由相移图片计算纹理图片
 *
 * @param imgs           相移图片(CV_8UC1)
 * @param texture        纹理图片(CV_8UC1)
 * @param phaseShiftStep 相移步数（@note 前N张图片为相移图片）
 * @param isColor        是否为彩色纹理
 * @param rowBegin       起始行
 * @param rowEnd         结束行
 */
void averageTextureRegionGrey(IN const std::vector<cv::Mat> &imgs,
                              OUT cv::Mat &texture, IN const int phaseShiftStep,
                              IN const int rowBegin, IN const int rowEnd);
/**
 * @brief                由相移图片计算纹理图片
 *
 * @param imgs           相移图片(CV_8UC3)
 * @param texture        纹理图片(CV_8UC3)
 * @param phaseShiftStep 相移步数（@note 前N张图片为相移图片）
 * @param isColor        是否为彩色纹理
 * @param rowBegin       起始行
 * @param rowEnd         结束行
 */
void averageTextureRegionColor(IN const std::vector<cv::Mat> &imgs,
                               OUT cv::Mat &texture,
                               IN const int phaseShiftStep,
                               IN const int rowBegin, IN const int rowEnd);

/**
 * @brief 反向映射计算深度图
 * @param depth         深度图(CV_32FC1)
 * @param textureIn     纹理图(CV_8UC3)
 * @param info          标定信息
 * @param textureAlign  对齐到深度图的纹理图(CV_8UC3)
 */
void reverseMappingTexture(IN const cv::Mat &depth, IN const cv::Mat &textureIn,
                           IN const Info &info, OUT cv::Mat &textureAlign);
/**
 * @brief
 * @param depth         输入，深度图(CV_32FC1)
 * @param textureIn     输入，纹理图(CV_8UC3)
 * @param info          输入，标定信息
 * @param textureAlign  输出，对齐到深度图的纹理图(CV_8UC3)
 * @param rowBegin      输入，起始行
 * @param rowEnd        输入，结束行
 */
void reverseMappingTextureRegion(IN const cv::Mat &depth,
                                 IN const cv::Mat &textureIn,
                                 IN const Info &info, OUT cv::Mat &textureAlign,
                                 IN const int rowBegin, IN const int rowEnd);
/**
 * @brief 过滤相位
 *
 * @param absPhase 需要过滤的绝对相位
 * @param out 过滤后的绝对相位
 * @param maxTollerance 允许最大不同量
 * @param kenel 卷积核大小
 */
void filterPhase(IN const cv::Mat &absPhase, OUT cv::Mat &out,
                 IN const float maxTollerance = 0.5, IN const int kernel = 5);
/**
 * @brief 过滤绝对相位入口函数
 *
 * @param absPhase 需要过滤的绝对相位
 * @param out 过滤后的绝对相位
 * @param maxTollerance 允许最大不同量
 * @param rowBeign 起始行
 * @param rowEnd 结束行
 * @param kenel 卷积核大小
 */
void entryFilterPhase(IN const cv::Mat &absPhase, OUT cv::Mat &out,
                      IN const int rowBeign, IN const int rowEnd,
                      IN const float maxTollerance = 0.5, IN int kernel = 5);
/**
 * @brief 将校正后的相机坐标系下深度图映射回校正前并获得点云
 * @param depthIn 校正后相机坐标系下的深度图
 * @param textureIn 校正后相机坐标系下的纹理图
 * @param Q 校正后的深度映射矩阵
 * @param M 深度相机的内参
 * @param R1 校正矩阵
 * @param depthRemaped 校正前相机坐标系下的深度图
 * @param cloud 点云
 */
void remapToDepthCamera(IN const cv::Mat &depthIn, IN const cv::Mat &textureIn,
                        IN const cv::Mat &Q, IN const cv::Mat &M,
                        IN const cv::Mat &R1, OUT cv::Mat &depthRemaped,
                        OUT pcl::PointCloud<pcl::PointXYZRGB> &cloud);
} // namespace tool
} // namespace sl

#endif // TOOL_TOOL_H_
