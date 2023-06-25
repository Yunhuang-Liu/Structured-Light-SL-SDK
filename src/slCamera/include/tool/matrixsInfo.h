/**
 * @file matrixsInfo.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief  标定信息工具类
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef __TOOL_MATRIXINFO_H_
#define __TOOL_MATRIXINFO_H_

#include <string>

#include <opencv2/opencv.hpp>

#include <typeDef.h>

/** @brief 结构光库 */
namespace sl {
/** @brief 工具库 */
namespace tool {
/** @brief 信息结构体    */
struct Info {
    /** \左相机内参矩阵(CV_64FC1) **/
    cv::Mat __M1;
    /** \右相机内参矩阵(CV_64FC1) **/
    cv::Mat __M2;
    /** \彩色相机内参矩阵(CV_64FC1) **/
    cv::Mat __M3;
    /** \投影仪内参矩阵(CV_64FC1) **/
    cv::Mat __M4;
    /** \左相机相机坐标系到右相机坐标系的旋转矩阵(CV_64FC1) **/
    cv::Mat __R1;
    /** \右相机相机坐标系到左相机坐标系的旋转矩阵(CV_64FC1) **/
    cv::Mat __R2;
    /** \左相机相机坐标系到右相机的投影矩阵(CV_64FC1) **/
    cv::Mat __P1;
    /** \右相机相机坐标系到世界坐标系的投影矩阵(CV_64FC1) **/
    cv::Mat __P2;
    /** \左相机的畸变矩阵(CV_64FC1) **/
    cv::Mat __D1;
    /** \右相机的畸变矩阵(CV_64FC1) **/
    cv::Mat __D2;
    /** \彩色相机的畸变矩阵(CV_64FC1) **/
    cv::Mat __D3;
    /** \投影仪的畸变矩阵(CV_64FC1) **/
    cv::Mat __D4;
    /** \深度映射矩阵(CV_64FC1) **/
    cv::Mat __Q;
    /** \左相机八参数矩阵(CV_64FC1) **/
    cv::Mat __K1;
    /** \右相机八参数矩阵(CV_64FC1) **/
    cv::Mat __K2;
    /** \左相机至右相机旋转矩阵(CV_64FC1) **/
    cv::Mat __Rlr;
    /** \左相机至右相机平移矩阵(CV_64FC1) **/
    cv::Mat __Tlr;
    /** \左相机至投影仪旋转矩阵(CV_64FC1) **/
    cv::Mat __Rlp;
    /** \左相机至投影仪平移矩阵(CV_64FC1) **/
    cv::Mat __Tlp;
    /** \右相机至投影仪旋转矩阵(CV_64FC1) **/
    cv::Mat __Rrp;
    /** \右相机至投影仪平移矩阵(CV_64FC1) **/
    cv::Mat __Trp;
    /** \左相机至右相机本质矩阵(CV_64FC1) **/
    cv::Mat __E;
    /** \左相机至右相机本质矩阵(CV_64FC1) **/
    cv::Mat __F;
    /** \左相机至彩色相机旋转矩阵(CV_64FC1) **/
    cv::Mat __Rlc;
    /** \左相机至彩色相机平移矩阵(CV_64FC1) **/
    cv::Mat __Tlc;
    /** \相机幅面(CV_64FC1) **/
    cv::Mat __S;
};

/** @brief 相机标定信息类 */
class MatrixsInfo {
  public:
    /**
     * @brief 类的初始化，完成内外参数信息的导入
     *
     * @param intrinsicsPath 内参文件路径
     * @param extrinsicsPath 外参文件路径
     */
    MatrixsInfo(IN const std::string intrinsicsPath,
                IN const std::string extrinsicsPath);
    /**
     * @brief 类的初始化，完成内外参数信息的导入
     *
     * @param intrinsicsPath 系统参数文件路径
     */
    MatrixsInfo(IN const std::string calibrationFileDir);
    /**
     * @brief 获取标定信息
     *
     * @return const Info& 信息结构体
     */
    const Info &getInfo();

  private:
    /** \读取到的校正信息 **/
    Info __info;
};
} // namespace tool
} // namespace sl
#endif // !__TOOL_MATRIXINFO_H_