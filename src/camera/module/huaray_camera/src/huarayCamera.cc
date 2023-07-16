#include "huarayCamera.h"

#include <chrono>

namespace sl {
namespace camera {

//相机取流回调函数
static void frameCallback(IMV_Frame *pFrame, void *pUser) {
    HuarayCammera *pCamera = reinterpret_cast<HuarayCammera *>(pUser);
    if (pFrame->pData != NULL) {
        //TODO@LiuYunhuang:增加各种格式的pack支持
        std::string pixelSize;
        pCamera->getEnumAttribute("PixelSize", pixelSize);

        cv::Mat img;
        if("Bpp8" == pixelSize) {
            img = cv::Mat(pFrame->frameInfo.height, pFrame->frameInfo.width,
                                CV_8U, pFrame->pData)
                            .clone();
        }
        else if("Bpp16" == pixelSize) {
            img = cv::Mat(pFrame->frameInfo.height, pFrame->frameInfo.width,
                                CV_16U, pFrame->pData)
                            .clone();
        }
        else if("Bpp24" == pixelSize) {
            img = cv::Mat(pFrame->frameInfo.height, pFrame->frameInfo.width,
                                CV_8UC3, pFrame->pData)
                            .clone();
        }

        if(pCamera->getImgs().size() >= 50) {
            pCamera->popImg();    
        }

        pCamera->pushImg(img);
    }
}

HuarayCammera::HuarayCammera(const std::string cameraUserId)
    : __cameraUserId(cameraUserId), __pCamera(nullptr) {}

HuarayCammera::~HuarayCammera() {}

CameraInfo HuarayCammera::getCameraInfo() {
    CameraInfo info;
    info.__isFind = false;
    IMV_DeviceList deviceList;
    IMV_EnumDevices(&deviceList, IMV_EInterfaceType::interfaceTypeAll);
    for (size_t i = 0; i < deviceList.nDevNum; ++i) {
        if(__cameraUserId == deviceList.pDevInfo[i].cameraName) {
            info.__isFind = true;
            info.__cameraKey = deviceList.pDevInfo[i].cameraKey;
            info.__cameraUserId = deviceList.pDevInfo[i].cameraName;
            info.__deviceType = deviceList.pDevInfo[i].nInterfaceType;
        }
    }
    return info;
}

bool HuarayCammera::connect() {
    auto ret =
        IMV_CreateHandle((void**)&__pCamera, IMV_ECreateHandleMode::modeByDeviceUserID,
                        (void*)__cameraUserId.data());

    if (IMV_OK != ret) {
        printf("create devHandle failed! userId[%s], ErrorCode[%d]\n",
               __cameraUserId.data(), ret);
        return false;
    }

    ret = IMV_Open(__pCamera);
    if (IMV_OK != ret) {
        printf("open camera failed! ErrorCode[%d]\n", ret);
        return false;
    }

    return true;
}

bool HuarayCammera::disConnect() {
    if (!__pCamera) {
        printf("close camera fail. No camera.\n");
        return false;
    }

    if (false == IMV_IsOpen(__pCamera)) {
        printf("camera is already close.\n");
        return false;
    }

    auto ret = IMV_Close(__pCamera);
    if (IMV_OK != ret) {
        printf("close camera failed! ErrorCode[%d]\n", ret);
        return false;
    }

    ret = IMV_DestroyHandle(__pCamera);
    if (IMV_OK != ret) {
        printf("destroy devHandle failed! ErrorCode[%d]\n", ret);
        return false;
    }

    __pCamera = nullptr;

    return true;
}

std::queue<cv::Mat>& HuarayCammera::getImgs() {
    return __imgs;
}

bool HuarayCammera::pushImg(const cv::Mat &img) {
    __imgs.emplace(img);

    return true;
}

cv::Mat HuarayCammera::popImg() {
    cv::Mat img = __imgs.front();
    __imgs.pop();

    return img;
}

bool HuarayCammera::clearImgs() {
    std::queue<cv::Mat> emptyQueue;
    __imgs.swap(emptyQueue);

    return true;
}

bool HuarayCammera::isConnect() { return IMV_IsOpen(__pCamera); }

cv::Mat HuarayCammera::capture() {
    if (!__pCamera) {
        printf("Error, camera dosn't open! \n");
        return cv::Mat();
    }

    const int preNums = __imgs.size();

    setTrigMode(TrigMode::trigSoftware);

    auto ret = IMV_ExecuteCommandFeature(__pCamera, "TriggerSoftware");
    if (IMV_OK != ret) {
        printf("ExecuteSoftTrig fail, ErrorCode[%d]\n", ret);
        return cv::Mat();
    }

    double exposureTime;
    getNumbericalAttribute("ExposureTime", exposureTime);
    auto timeBegin = std::chrono::system_clock::now();
    while(preNums == __imgs.size()) {
        auto timeEnd = std::chrono::system_clock::now();
        auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeBegin).count()* (double)std::chrono::milliseconds::period::num / std::chrono::milliseconds::period::den;
        if(timeElapsed > (exposureTime / 1000000.0 * 2))  {
            break;
        }
    };

    cv::Mat softWareCapturedImg = __imgs.back();
    return softWareCapturedImg;
}

bool HuarayCammera::start() {
    if (!__pCamera) {
        printf("Error, camera dosn't open! \n");
        return false;
    }

    if (IMV_IsGrabbing(__pCamera)) {
        printf("camera is already grebbing.\n");
        return false;
    }

    auto ret = IMV_AttachGrabbing(__pCamera, frameCallback, this);

    if (IMV_OK != ret) {
        printf("Attach grabbing failed! ErrorCode[%d]\n", ret);
        return false;
    }

    ret = IMV_StartGrabbing(__pCamera);
    if (IMV_OK != ret) {
        printf("start grabbing failed! ErrorCode[%d]\n", ret);
        return false;
    }

    return true;
}

bool HuarayCammera::pause() {
    if (!__pCamera) {
        printf("Error, camera dosn't open! \n");
        return false;
    }

    if (!IMV_IsGrabbing(__pCamera)) {
        printf("camera is already stop grubbing.\n");
        return false;
    }

    auto ret = IMV_StopGrabbing(__pCamera);
    if (IMV_OK != ret) {
        printf("Stop grubbing failed! ErrorCode[%d]\n", ret);
        return false;
    }

    return true;
}

bool HuarayCammera::setTrigMode(const TrigMode trigMode) {
    if (!__pCamera) {
        printf("Error, camera dosn't open! \n");
        return false;
    }

    int ret = IMV_OK;
    if (trigContinous == trigMode) {
        ret = IMV_SetEnumFeatureSymbol(__pCamera, "TriggerMode", "Off");
        if (IMV_OK != ret) {
            printf("set TriggerMode value = Off fail, ErrorCode[%d]\n", ret);
            return false;
        }
    } else if (trigSoftware == trigMode) {
        ret = IMV_SetEnumFeatureSymbol(__pCamera, "TriggerMode", "On");
        if (IMV_OK != ret) {
            printf("set TriggerMode value = On fail, ErrorCode[%d]\n", ret);
            return false;
        }

        ret = IMV_SetEnumFeatureSymbol(__pCamera, "TriggerSource", "Software");
        if (IMV_OK != ret) {
            printf("set TriggerSource value = Software fail, ErrorCode[%d]\n",
                   ret);
            return false;
        }
    } else if (trigLine == trigMode) {
        ret = IMV_SetEnumFeatureSymbol(__pCamera, "TriggerMode", "On");
        if (IMV_OK != ret) {
            printf("set TriggerMode value = On fail, ErrorCode[%d]\n", ret);
            return false;
        }

        ret = IMV_SetEnumFeatureSymbol(__pCamera, "TriggerSource", "Line2");
        if (IMV_OK != ret) {
            printf("set TriggerSource value = Line1 fail, ErrorCode[%d]\n",
                   ret);
            return false;
        }
    }
    return true;
}

bool HuarayCammera::setEnumAttribute(const std::string attributeName,
                                     const std::string val) {
    if (!__pCamera) {
        printf("Error, camera dosn't open! \n");
        return false;
    }

    return IMV_SetEnumFeatureSymbol(__pCamera, attributeName.data(),
                                    val.data()) == IMV_OK;
}

bool HuarayCammera::setStringAttribute(const std::string attributeName,
                                       const std::string val) {
    if (!__pCamera) {
        printf("Error, camera dosn't open! \n");
        return false;
    }

    return IMV_SetStringFeatureValue(__pCamera, attributeName.data(),
                                     val.data()) == IMV_OK;
}

bool HuarayCammera::setNumberAttribute(const std::string attributeName,
                                           const double val) {
    if (!__pCamera) {
        printf("Error, camera dosn't open! \n");
        return false;
    }

    return IMV_SetDoubleFeatureValue(__pCamera, attributeName.data(), val) == IMV_OK;
}

bool HuarayCammera::setBooleanAttribute(const std::string attributeName,
                                        const bool val) {
    if (!__pCamera) {
        printf("Error, camera dosn't open! \n");
        return false;
    }

    return IMV_SetBoolFeatureValue(__pCamera, attributeName.data(), val) == IMV_OK;
}

bool HuarayCammera::getEnumAttribute(const std::string attributeName,
                                     std::string &val) {
    if (!__pCamera) {
        printf("Error, camera dosn't open! \n");
        return false;
    }

    IMV_String data;
    IMV_GetEnumFeatureSymbol(__pCamera, attributeName.data(), &data);
    val = data.str;

    return true;
}

bool HuarayCammera::getStringAttribute(const std::string attributeName,
                                       std::string &val) {
    if (!__pCamera) {
        printf("Error, camera dosn't open! \n");
        return false;
    }

    IMV_String data;
    IMV_GetStringFeatureValue(__pCamera, attributeName.data(), &data);
    val = data.str;

    return true;
}

bool HuarayCammera::getNumbericalAttribute(const std::string attributeName,
                                           double &val) {
    if (!__pCamera) {
        printf("Error, camera dosn't open! \n");
        return false;
    }

    IMV_GetDoubleFeatureValue(__pCamera, attributeName.data(), &val);

    return true;
}

bool HuarayCammera::getBooleanAttribute(const std::string attributeName,
                                        bool &val) {
    if (!__pCamera) {
        printf("Error, camera dosn't open! \n");
        return false;
    }

    IMV_GetBoolFeatureValue(__pCamera, attributeName.data(), &val);

    return true;
}
} // namespace camera
} // namespace sl