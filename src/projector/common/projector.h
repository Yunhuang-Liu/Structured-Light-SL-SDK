#ifndef __PROJECTOR_H_
#define __PROJECTOR_H_

#include "typeDef.h"

#include <opencv2/opencv.hpp>

/** @brief 结构光相机库 */
namespace sl {
/** @brief 投影仪库 */
namespace projector {
/** @brief 投影仪LED灯 */
enum Illumination { Red = 0, Grren, Blue, RGB };

/** @brief 投影图案集合 */
struct PatternOrderSet {
    std::vector<cv::Mat> __imgs; //需要制作集合的图片
    // int __patternSetIndex;        //集合索引
    int __patternArrayCounts;    //图片数组数量
    Illumination __illumination; // LED控制
    bool __invertPatterns;       //反转图片
    bool __isVertical;           //是否水平图片
    bool __isOneBit;             //是否为一位深度
    int __exposureTime;          //曝光时间(us)
    int __preExposureTime;       //曝光前时间(us)
    int __postExposureTime;      //曝光后时间(us)
};

/** @brief 相机信息 **/
struct ProjectorInfo {
    std::string __dlpEvmType; // DLP评估模块
    int __width;              //幅面宽度
    int __height;             //幅面高度
    bool __isFind;            //是否找到
};

/** @brief 投影仪控制类 */
class Projector {
  public:
    virtual ~Projector() {}
    /**
     * @brief 获取投影仪信息
     *
     * @return ProjectorInfo 投影仪相关信息
     */
    virtual ProjectorInfo getInfo() = 0;
    /**
     * @brief 连接
     *
     * @return true 成功
     * @return false 失败
     */
    virtual bool connect() = 0;
    /**
     * @brief 断开连接
     *
     * @return true 成功
     * @return false 失败
     */
    virtual bool disConnect() = 0;
    /**
     * @brief 断开连接
     *
     * @return true 成功
     * @return false 失败
     */
    virtual bool isConnect() = 0;
    /**
     * @brief 从图案集制作投影序列
     *
     * @param table 投影图案集
     */
    virtual bool
    populatePatternTableData(IN std::vector<PatternOrderSet> table) = 0;
    /**
     * @brief 投影
     *
     * @param isContinue 是否连续投影
     * @return true 成功
     * @return false 失败
     */
    virtual bool project(IN const bool isContinue) = 0;
    /**
     * @brief 暂停
     *
     * @return true 成功
     * @return false 失败
     */
    virtual bool pause() = 0;
    /**
     * @brief 停止
     *
     * @return true 成功
     * @return false 失败
     */
    virtual bool stop() = 0;
    /**
     * @brief 恢复投影
     *
     * @return true 成功
     * @return false 失败
     */
    virtual bool resume() = 0;
    /**
     * @brief 投影下一帧
     * @warning 仅在步进模式下使用
     *
     * @return true
     * @return false
     */
    virtual bool step() = 0;
    /**
     * @brief 获取当前LED三色灯电流值
     *
     * @param r 红色电流值
     * @param g 绿色电流值
     * @param b 蓝色电流值
     * @return true 成功
     * @return false 失败
     */
    virtual bool getLEDCurrent(OUT double &r, OUT double &g, OUT double &b) = 0;
    /**
     * @brief 设置当前LED三色灯电流值
     *
     * @param r 红色电流值
     * @param g 绿色电流值
     * @param b 蓝色电流值
     * @return true 成功
     * @return false 失败
     */
    virtual bool setLEDCurrent(IN const double r, IN const double g,
                               IN const double b) = 0;
    /**
     * @brief 获取当前闪存图片数量
     *
     * @return int 图片数量
     */
    virtual int getFlashImgsNum() = 0;
  private:
};
} // namespace projector
} // namespace sl

#endif //!__PROJECTOR_H_
