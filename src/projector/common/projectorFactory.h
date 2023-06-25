#ifndef __PROJECTORY_FACTORY_H_
#define __PROJECTORY_FACTORY_H_

#include <string>
#include <unordered_map>

#include "projector.h"
#include "projectorDlpc34xx.h"
#include "projectorDlpc34xxDual.h"

/** @brief 结构光库 **/
namespace sl {
/** @brief 投影仪库 **/
namespace projector {
/** @brief 投影仪工厂 **/
class ProjectorFactory {
  public:
    ProjectorFactory() { };

    Projector *getProjector(const std::string dlpEvm) {
        Projector *projector = nullptr;

        if (__projectoies.count(dlpEvm)) {
            return __projectoies[dlpEvm];
        } else {
            if ("DLP4710" == dlpEvm) {
                projector = new ProjectorDlpc34xxDual();
                __projectoies[dlpEvm] = projector;
            }
            
            else if ("DLP3010" == dlpEvm) {
                projector = new ProjectorDlpc34xx();
                __projectoies[dlpEvm] = projector;
            }
            //TODO@LiuYunhuang:增加DLP6500支持
        }

        return projector;
    }
  private:
    std::unordered_map<std::string, Projector *> __projectoies;
}; // class CameraFactory
} // namespace camera
} // namespace sl

#endif //__PROJECTORY_FACTORY_H_