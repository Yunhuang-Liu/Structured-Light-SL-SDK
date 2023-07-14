#ifndef __BINOCULAR_CAMERA_H_
#define __BINOCULAR_CAMERA_H_

#include <slCamera.h>

#include <cameraFactory.h>
#include <matrixsInfo.h>
#include <nStepNGrayCodeMasterCpu.h>
#include <nShiftLineNGrayCodeMasterCpu.h>
#include <projectorFactory.h>
#include <rectifierCpu.h>
#include <restructorCpu.h>
#include <restructorShiftLineCpu.h>
#include <codeMaker.h>
#ifdef __WITH_CUDA__
#include <nStepNGrayCodeMasterGpu.h>
#include <rectifierGpu.h>
#include <restructorGpu.h>
#endif //!__WITH_CUDA__

#include <unordered_map>

/** @brief 结构光相机库 */
namespace sl {
/** @brief 3D相机库 */
namespace slCamera {
/** @brief 结构光相机 */
class BinocularCamera : public SLCamera {
  public:
    /**
     * @brief 使用配置文件加载相机配置
     *
     * @param jsonPath json文件路径
     */
    BinocularCamera(IN const std::string jsonPath);
    /**
     * @brief 获取相机信息
     *
     * @return SLCameraInfo 相机信息
     */
    SLCameraInfo getCameraInfo() override;
    /**
     * @brief 连接
     *
     * @return true 成功
     * @return false 失败
     */
    bool connect() override;
    /**
     * @brief 断开连接
     *
     * @return true 成功
     * @return false 失败
     */
    bool disConnect() override;
    /**
     * @brief 查询连接状态
     *
     * @return true 已连接
     * @return false 已断连
     */
    bool isConnect() override;
    /**
     * @brief 获取一帧数据
     *
     * @param frameData 获取到的数据
     * @return true 成功
     * @return false 失败
     */
    bool capture(IN FrameData &frameData) override;
    /**
     * @brief 获取一帧数据(离线)
     *
     * @param leftImgs 输入的左相机图片
     * @param rightImgs 输入的右相机图片
     * @param frameData 获取到的数据
     * @return true 成功
     * @return false 失败
     */
    bool offLineCapture(IN const std::vector<cv::Mat>& leftImgs, IN const std::vector<cv::Mat>& rightImgs, OUT FrameData &frameData) override;
    /**
     * @brief 是否使能深度相机
     *
     * @param isEnable 使能标志位
     * @return true 成功
     * @return false 失败
     */
    bool setDepthCameraEnabled(IN const bool isEnable) override;
    /**
     * @brief 获取字符属性值
     *
     * @param attributeName 字符属性名称
     * @param val 当前字符属性值
     * @return true 成功
     * @return false 失败
     */
    bool getStringAttribute(IN const std::string attributeName,
                            OUT std::string &val) override;
    /**
     * @brief 获取数字属性值
     *
     * @param attributeName 数字属性名称
     * @param val 当前数字属性值
     * @return true 成功
     * @return false 失败
     */
    bool getNumbericalAttribute(IN const std::string attributeName,
                                OUT double &val) override;
    /**
     * @brief 获取布尔属性值
     *
     * @param attributeName 布尔属性名称
     * @param val 当前布尔属性值
     * @return true 成功
     * @return false 失败
     */
    bool getBooleanAttribute(IN const std::string attributeName,
                             OUT bool &val) override;
    /**
     * @brief 设置字符属性值
     *
     * @param attributeName 字符属性名称
     * @param val 需要设置的值
     * @return true 成功
     * @return false 失败
     */
    bool setStringAttribute(IN const std::string attributeName,
                            IN const std::string val) override;
    /**
     * @brief 设置数字属性值
     *
     * @param attributeName 数字属性名称
     * @param val 需要设置的值
     * @return true 成功
     * @return false 失败
     */
    bool setNumberAttribute(IN const std::string attributeName,
                            IN const double val) override;
    /**
     * @brief 设置布尔属性值
     *
     * @param attributeName 布尔属性名称
     * @param val 需要设置的值
     * @return true 成功
     * @return false 失败
     */
    bool setBooleanAttribute(IN const std::string attributeName,
                             IN const bool val) override;
    /**
     * @brief 重置相机配置
     *
     * @return true 成功
     * @return false 失败
     */
    bool resetCameraConfig() override;
    /**
     * @brief 更新相机
     *
     * @return true 成功
     * @return false 失败
     */
    bool updateCamera() override;
  private:
    /**
     * @brief 读取配置文件
     *
     * @param jsonPath 配置文件路径
     * @param jsonVal 配置值
     * @return true 成功
     * @return false 失败
     */
    bool loadParams(IN const std::string jsonPath, IN Json::Value &jsonVal);
    /**
     * @brief 写入配置文件
     *
     * @param jsonPath 配置文件路径
     * @param jsonVal 配置
     * @return true 成功
     * @return false 失败
     */
    bool saveParams(IN const std::string jsonPath, IN Json::Value &jsonVal);
    /**
     * @brief 解析数组
     *
     * @param jsonVal 配置值
     * @param isWrite 是否写入
     */
    void parseArray(IN Json::Value &jsonVal, IN const bool isWrite);
    /**
     * @brief 更新加速方法
     *
     */
    void updateAlgorithmMethod();
    /**
     * @brief 更新曝光时间
     * 
     */
    void updateExposureTime();
    /**
     * @brief 更新闪存图片
     * 
     */
    void updateFlashPattern();
    /**
     * @brief 更新深度相机使能
     * 
     */
    void updateEnableDepthCamera();
    /**
     * @brief 更新投影仪亮度
     * 
     */
    void updateLightStrength();
    bool __isInitial;
    std::string __jsonPath;
    Json::Value __jsonVal;
    SLCameraInfo __cameraInfo;
    std::unordered_map<std::string, std::string> __stringProperties;
    std::unordered_map<std::string, float> __numbericalProperties;
    std::unordered_map<std::string, bool> __booleanProperties;
    std::unordered_map<std::string, bool> __propertiesChangedSignals;
    camera::CameraFactory __cameraFactory;
    projector::ProjectorFactory __projectorFactory;
    tool::MatrixsInfo *__matrixInfo;
    phaseSolver::PhaseSolver *__phaseSolver;
    rectifier::Rectifier *__rectifier;
    restructor::Restructor *__restructor;
};
} // namespace slCamera
} // namespace sl

#endif // !__BINOCULAR_CAMERA_H_