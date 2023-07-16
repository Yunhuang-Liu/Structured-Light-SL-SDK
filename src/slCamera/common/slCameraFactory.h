#ifndef __SL_CAMERA_FACTORY_H_
#define __SL_CAMERA_FACTORY_H_

#include <string>
#include <unordered_map>

#include "slCamera.h"
#include "binoocularCamera.h"

/** @brief 结构光库 **/
namespace sl {
/** @brief 结构光相机库 **/
namespace slCamera {
/** @brief 相机工厂 **/
class SLCameraFactory {
  public:
    SLCameraFactory() { };
    /**@brief 制造商*/
    enum CameraManufactor {
        BinoocularCamera = 0, //华睿科技
        MonocularCamera       //海康机器人
    };

    SLCamera *getCamera(std::string cameraJsonConfig,
                            CameraManufactor manufactor) {
        SLCamera *camera = nullptr;

        if (__cameras.count(cameraJsonConfig)) {
            return __cameras[cameraJsonConfig];
        } else {
            if (BinoocularCamera == manufactor) {
                camera = new BinocularCamera(cameraJsonConfig);
                __cameras[cameraJsonConfig] = camera;
            }
            // TODO@LiuYunhuang:增加海康相机支持
            else if (MonocularCamera == manufactor) {
                camera = new BinocularCamera(cameraJsonConfig);
                __cameras[cameraJsonConfig] = camera;
            }
        }

        return camera;
    }
  private:
    std::unordered_map<std::string, SLCamera *> __cameras;
}; // class CameraFactory
} // namespace camera
} // namespace sl

#endif //__SL_CAMERA_FACTORY_H_