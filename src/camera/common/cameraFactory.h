#ifndef __CAMERA_FACTORY_H_
#define __CAMERA_FACTORY_H_

#include <string>
#include <unordered_map>

#include "camera.h"
#include "huarayCamera.h"

/** @brief 结构光库 **/
namespace sl {
/** @brief 相机库 **/
namespace camera {
/** @brief 相机工厂 **/
class CameraFactory {
  public:
    CameraFactory() { };
    /**@brief 制造商*/
    enum CameraManufactor {
        Huaray = 0, //华睿科技
        Halcon      //海康机器人
    };

    Camera *getCamera(std::string cameraUserId,
                            CameraManufactor manufactor) {
        Camera *camera = nullptr;

        if (__cameras.count(cameraUserId)) {
            return __cameras[cameraUserId];
        } else {
            if (Huaray == manufactor) {
                camera = new HuarayCammera(cameraUserId);
                __cameras[cameraUserId] = camera;
            }
            // TODO@LiuYunhuang:增加海康相机支持
            else if (Halcon == manufactor) {
                camera = new HuarayCammera(cameraUserId);
                __cameras[cameraUserId] = camera;
            }
        }

        return camera;
    }
  private:
    std::unordered_map<std::string, Camera *> __cameras;
}; // class CameraFactory
} // namespace camera
} // namespace sl

#endif //__CAMERA_FACTORY_H_