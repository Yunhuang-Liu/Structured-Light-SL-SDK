/**
 * @file CodeMaker.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2021-10-15
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <opencv2/opencv.hpp>

namespace elencode {
	cv::Mat createBWShiftCode(const int width, const int height, const int blockNum, const int shiftTime);
}

/** \格雷码编码器 **/
class GrayEncoder {
  public:
    /**
     * @brief 显式构造器，禁止隐式转换
     *
     * @param encodeSize 输入，编码图像尺寸(宽*高)
     */
    explicit GrayEncoder(const cv::Size encodeSize);
    /**
     * @brief 获取生成的格雷码图像
     *
     * @return cv::Mat 输出，生成的格雷码图像
     */
    cv::Mat getImage();
    /**
     * @brief 创建格雷码图像
     * 		** \此处设计存在揣摩:接口应当尽量少的暴露给客户，但类应当维护自身的资源
     * /**
     *
     * @param numOfImage 输入，图片数量
     * @return cv::Mat 输出，生成的图片
     */
    cv::Mat creatGrayImage(const int numOfImage, const int imgPlainNum,
                           const int grayBits = 4);
    /**
     * @brief 设置位移格雷码标志位
     * @param state 输入，状态位
     */
    void setShiftGrayCodeState(const bool state);
    /**
     * @brief 设置竖向条纹标志位
     *
     * @param state 输入，状态位
     */
    void setChangeModeY(const bool state);
    /**
     * @param 设置1Bit图片生成标志位
     *
     * @state 输入，状态位
     */
    void setIsOneHeightMode(const bool state);
    /**
     * @param 设置第一位取反模式
     *
     * @state 输入，状态位
     */
    void setCounterFirstBit(const bool state);
    /**
     * @param 设置四灰度格雷编码状态
     *
     * @state 输入，状态位
     */
    void setFourFloorMode(const bool state);

  private:
    /** \图片尺寸 **/
    cv::Size imgSize;
    /** \生成的格雷码图片 **/
    cv::Mat myGrayEncodeImage;
    /** \生成的位移格雷码图片 **/
    cv::Mat shiftGrayCodeImage;
    /** \位移格雷码标志位 **/
    bool isShiftGrayCode;
    /** \是否生成竖向条纹 **/
    bool isChangeModeY;
    /** \是否生成1高度图片 **/
    bool isOneHeight;
    /** \是否第一位取反 **/
    bool isCounterFirstBit;
    /** \四灰度编码状态位 **/
    bool isFourFloorGray;
};

/** \正弦相移光栅编码器 此编码器编码8bit图像，适配DLP3100投影仪**/
class PhaseEncoder {
  public:
    /**
     * @brief 显式构造器，禁止隐式转换
     *
     * @param encodeSize 输入，编码图像尺寸(宽*高)
     */
    explicit PhaseEncoder(const cv::Size);
    /**
     * @brief 获取生成的相移光栅图像
     *
     * @return const cv::Mat 输出，生成的格雷码图像
     */
    std::vector<cv::Mat>& getImage();
    /**
     * @brief 创建相移光栅图像
     *
     * @param numOfPhaseShift 输入，相移步数
     * @param patchDistance 输入，莫尔条纹节距
     * @param 输出，创建的相移图片
     */
    std::vector<cv::Mat>& creatPhaseImage(const int numOfPhaseShift,
                            const float patchDistance);
    /**
     * @brief 设置二元光栅标志位
     *
     * @param state 输入，标志位
     */
    void setBinaryPhaseCode(const bool state);
    /**
     * @brief 设置OPWM光栅标志位
     *
     * @param state 输入，标志位
     */
    void setOPWMCode(const bool state);
    /**
     * @brief 设置反向相移模式标志位
     *
     * @param state 输入，标志位
     */
    void setCounterMode(const bool state);
    /**
     * @brief 设置纯白模式标志位
     *
     * @param state 输入，标志位
     */
    void setWhiteMode(const bool state);
    /**
     * @brief 设置二维抖动标志位
     *
     * @param state 输入，标志位
     */
    void setPhaseErrorExpandMode(const bool state);
    /**
     * @brief 生成理想正弦图片
     *
     * @param shiftNum 输入，相移步数
     * @param imgs 输入，生成的相移图片
     */
    void createImgeableImg(const int shiftNum, const int patchDistance,
                           std::vector<cv::Mat> &imgs);
    /**
     * @brief 设置竖直条纹生成标志位
     *
     * @param state 输入，标志位
     */
    void setChangeModeY(const bool state);
    /**
     * @param 设置1Bit图片生成标志位
     *
     * @state 输入，状态位
     */
    void setIsOneHeightMode(const bool state);
    /**
     * @param 设置反方向位移模式
     *
     * @state 输入，状态位
     */
    void setCounterDirection(const bool state);

  private:
    /** \图像尺寸 **/
    cv::Size imgSize;
    /** \生成的相移光栅图片 **/
    std::vector<cv::Mat> myPhaseEncodeImage;
    /** \离焦投影生成二元光栅标志位 **/
    bool isBinaryPhaseCoder;
    /** \离焦投影生成OPWM光栅标志位 **/
    bool isOPWMCoder;
    /** \反向相移模式标志位 **/
    bool isCounterMode;
    /** \全白图片 **/
    bool isWhite;
    /** \二维抖动 **/
    bool isPhaseErrorExpand;
    /** \竖直条纹 **/
    bool isChangeModeY;
    /** \是否生成1Bit图片 **/
    bool isOneHeight;
    /** \是否反方向 **/
    bool isCounterDirection;
};
