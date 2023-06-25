#include <binoocularCamera.h>

#include <tool.h>

using namespace sl::camera;
using namespace sl::projector;
using namespace sl::phaseSolver;
using namespace sl::tool;
using namespace sl::restructor;
using namespace sl::rectifier;

/** @brief 结构光相机库 */
namespace sl {
/** @brief 3D相机库 */
namespace slCamera {
BinocularCamera::BinocularCamera(IN const std::string jsonPath)
    : SLCamera(jsonPath), __jsonPath(jsonPath), __matrixInfo(nullptr),
      __phaseSolver(nullptr), __rectifier(nullptr), __restructor(nullptr),
      __isInitial(false) {

    if (loadParams(jsonPath, __jsonVal)) {
        if (__booleanProperties["Gpu Accelerate"]) {
#ifndef __WITH_CUDA__
            printf("lib isn't build with cuda, we will disable it! \n");
            __booleanProperties["Gpu Accelerate"] = false;
#endif //!__WITH_CUDA__
        }

        updateAccelerateMethod();

        __isInitial = true;
    }
}

bool BinocularCamera::loadParams(const std::string jsonPath,
                                 Json::Value &jsonVal) {
    if (!readJsonFile(jsonPath, jsonVal)) {
        printf("binocular camera parse json file error! \n");
        return false;
    }

    parseArray(jsonVal["camera"]["info"], false);
    parseArray(jsonVal["camera"]["preProcess"], false);
    parseArray(jsonVal["camera"]["afterProcess"], false);

    return true;
}

bool BinocularCamera::saveParams(const std::string jsonPath,
                                 Json::Value &jsonVal) {
    if (jsonVal.empty()) {
        printf(
            "binocular camera write json file error, json value is empty! \n");
        return false;
    }

    parseArray(jsonVal["camera"]["info"], true);
    parseArray(jsonVal["camera"]["preProcess"], true);
    parseArray(jsonVal["camera"]["afterProcess"], true);

    std::ofstream writeStream(jsonPath);
    Json::StyledWriter writer;
    writeStream << writer.write(jsonVal);

    return true;
}

void BinocularCamera::parseArray(Json::Value &jsonVal, const bool isWrite) {
    const int numOfCameraInfo = jsonVal.size();

    for (int i = 0; i < numOfCameraInfo; ++i) {
        const std::string titleString = jsonVal[i]["type"].asString();
        if (titleString == "string" || titleString == "enum") {
            if (!isWrite) {
                setStringAttribute(jsonVal[i]["title"].asString(),
                                   jsonVal[i]["data"].asString());
            } else {
                jsonVal[i]["data"] =
                    __stringProperties[jsonVal[i]["title"].asString()];
            }
        } else if (titleString == "number") {
            if (!isWrite) {
                setNumberAttribute(jsonVal[i]["title"].asString(),
                                   jsonVal[i]["data"].asDouble());
            } else {
                jsonVal[i]["data"] =
                    __numbericalProperties[jsonVal[i]["title"].asString()];
            }
        } else if (titleString == "bool") {
            if (!isWrite) {
                setBooleanAttribute(jsonVal[i]["title"].asString(),
                                    jsonVal[i]["data"].asBool());
            } else {
                jsonVal[i]["data"] =
                    (int)__booleanProperties[jsonVal[i]["title"].asString()];
            }
        }
    }
}

void BinocularCamera::updateAccelerateMethod() {
    if (__matrixInfo) {
        delete __matrixInfo;
        __matrixInfo = nullptr;
    }

    if (__phaseSolver) {
        delete __phaseSolver;
        __phaseSolver = nullptr;
    }

    if (__restructor) {
        delete __restructor;
        __restructor = nullptr;
    }

    if (__rectifier) {
        delete __rectifier;
        __rectifier = nullptr;
    }

    __matrixInfo = new MatrixsInfo(__stringProperties["Intrinsic Path"],
                                   __stringProperties["Extrinsic Path"]);

    if (__booleanProperties["Gpu Accelerate"]) {
#ifdef __WITH_CUDA__
        __rectifier = new RectifierCpu(__matrixInfo->getInfo());
        __phaseSolver = new NStepNGrayCodeMasterGpu();
        __restructor = new RestructorGpu(__matrixInfo->getInfo());
#endif //!__WITH_CUDA__
    } else {
        __rectifier = new RectifierCpu(__matrixInfo->getInfo());
        __phaseSolver = new NStepNGrayCodeMasterCpu();
        __restructor = new RestructorCpu(__matrixInfo->getInfo());
    }
}

SLCameraInfo BinocularCamera::getCameraInfo() {
    const CameraFactory::CameraManufactor manufator =
        __stringProperties["2D Camera Manufactor"] == "Huaray"
            ? CameraFactory::Huaray
            : CameraFactory::Halcon;
    auto pLeftCamera = __cameraFactory.getCamera(
        __stringProperties["Left Camera Name"], manufator);
    auto pRightCamera = __cameraFactory.getCamera(
        __stringProperties["Right Camera Name"], manufator);
    auto leftCameraInfo = pLeftCamera->getCameraInfo();
    auto rightCameraInfo = pRightCamera->getCameraInfo();

    auto pProjector =
        __projectorFactory.getProjector(__stringProperties["DLP Evm"]);
    auto projectorInfo = pProjector->getInfo();

    SLCameraInfo slCameraInfo;
    slCameraInfo.__isFind = leftCameraInfo.__isFind &&
                            rightCameraInfo.__isFind && projectorInfo.__isFind;
    slCameraInfo.__cameraName =
        slCameraInfo.__isFind ? __stringProperties["Camera Name"] : "NOT_FOUND";
    slCameraInfo.__intrinsic = slCameraInfo.__isFind
                                   ? __matrixInfo->getInfo().__M1
                                   : cv::Mat::zeros(3, 3, CV_32FC1);

    return slCameraInfo;
}

bool BinocularCamera::connect() {
    bool connectState = false;

    try {
        const CameraFactory::CameraManufactor manufator =
            __stringProperties["2D Camera Manufactor"] == "Huaray"
                ? CameraFactory::Huaray
                : CameraFactory::Halcon;
        const bool connectLeftCamera =
            __cameraFactory
                .getCamera(__stringProperties["Left Camera Name"], manufator)
                ->connect();
        const bool connectRightCamera =
            __cameraFactory
                .getCamera(__stringProperties["Right Camera Name"], manufator)
                ->connect();
        const bool connectProjector =
            __projectorFactory.getProjector(__stringProperties["DLP Evm"])
                ->connect();
        connectState =
            connectLeftCamera && connectRightCamera && connectProjector;

        if (connectState) {
            __cameraFactory
                .getCamera(__stringProperties["Left Camera Name"], manufator)
                ->start();
            __cameraFactory
                .getCamera(__stringProperties["Right Camera Name"], manufator)
                ->start();

            updateCamera();
        } else {
            if (__cameraFactory
                    .getCamera(__stringProperties["Left Camera Name"],
                               manufator)
                    ->isConnect()) {
                __cameraFactory
                    .getCamera(__stringProperties["Left Camera Name"],
                               manufator)
                    ->disConnect();
            }

            if (__cameraFactory
                    .getCamera(__stringProperties["Right Camera Name"],
                               manufator)
                    ->isConnect()) {
                __cameraFactory
                    .getCamera(__stringProperties["Right Camera Name"],
                               manufator)
                    ->disConnect();
            }
            if (__projectorFactory.getProjector(__stringProperties["DLP Evm"])
                    ->isConnect()) {
                __projectorFactory.getProjector(__stringProperties["DLP Evm"])
                    ->disConnect();
            }
        }
    } catch (...) {
        printf("Connect binocular camera error! \n");
        return false;
    }

    return connectState;
}

bool BinocularCamera::disConnect() {
    // TODO@LiuYunhuang:中途篡改不正确的2D相机制造商将导致漏洞，系统会重新查找该制造商同ID相机
    const CameraFactory::CameraManufactor manufator =
        __stringProperties["2D Camera Manufactor"] == "Huaray"
            ? CameraFactory::Huaray
            : CameraFactory::Halcon;
    const bool disConnectLeftCamera =
        __cameraFactory
            .getCamera(__stringProperties["Left Camera Name"], manufator)
            ->disConnect();
    const bool disConnectRightCamera =
        __cameraFactory
            .getCamera(__stringProperties["Right Camera Name"], manufator)
            ->disConnect();
    const bool disConnectProjector =
        __projectorFactory.getProjector(__stringProperties["DLP Evm"])
            ->disConnect();
    return disConnectLeftCamera && disConnectRightCamera && disConnectProjector;
}

bool BinocularCamera::isConnect() {
    const CameraFactory::CameraManufactor manufator =
        __stringProperties["2D Camera Manufactor"] == "Huaray"
            ? CameraFactory::Huaray
            : CameraFactory::Halcon;
    const bool isConnectLeftCamera =
        __cameraFactory
            .getCamera(__stringProperties["Left Camera Name"], manufator)
            ->isConnect();
    const bool isConnectRightCamera =
        __cameraFactory
            .getCamera(__stringProperties["Right Camera Name"], manufator)
            ->isConnect();
    const bool isConnectProjector =
        __projectorFactory.getProjector(__stringProperties["DLP Evm"])
            ->isConnect();

    return isConnectLeftCamera && isConnectRightCamera && isConnectProjector;
}

bool BinocularCamera::offLineCapture(const std::vector<cv::Mat> &leftImgs,
                                     const std::vector<cv::Mat> &rightImgs,
                                     FrameData &frameData) {
    std::vector<cv::Mat> leftImgsConvert(leftImgs.size()),
        rightImgConvert(rightImgs.size());
    // TODO@LiuYunhang:未用多线程加速，测试后视情况而定
    std::vector<cv::Mat> colorShiftImgs;
    for (size_t i = 0; i < leftImgs.size(); ++i) {
        if (leftImgs[i].type() == CV_8UC3) {
            colorShiftImgs.emplace_back(leftImgs[i]);
            cv::Mat grayImg;
            cv::cvtColor(leftImgs[i], grayImg, cv::COLOR_RGB2GRAY);
            __rectifier->remapImg(grayImg, leftImgsConvert[i], true);
        } else {
            __rectifier->remapImg(leftImgs[i], leftImgsConvert[i], true);
        }

        __rectifier->remapImg(rightImgs[i], rightImgConvert[i], false);
    }

    RestructParamater param;
    param.__minDisparity = __numbericalProperties["MinimumDisparity"];
    param.__maxDisparity = __numbericalProperties["MaximumDisparity"];
    param.__isMapToPreDepthAxes = false;
    param.__isMapToColorCamera = false;
    param.__minDepth = __numbericalProperties["MinimumDepth"];
    param.__maxDepth = __numbericalProperties["MaximumDepth"];
    param.__maximumCost = __numbericalProperties["MaximumMatchCost"];
#ifdef __WITH_CUDA__
    param.__block = dim3(__numbericalProperties["Num of block X"],
                         __numbericalProperties["Num of block Y"]);
#endif //!__WITH_CUDA__
    if (__booleanProperties["Gpu Accelerate"]) {
        PhaseSolverGroupDataDevice leftSovleData, rightSolveData;
        __phaseSolver->solve(leftImgsConvert, leftSovleData,
                             __numbericalProperties["Contrast Threshold"]);
        __phaseSolver->solve(rightImgConvert, rightSolveData,
                             __numbericalProperties["Contrast Threshold"]);

        cv::cuda::GpuMat filterLeftPhase, filterRightPhase, depthMap;
        if (__booleanProperties["Noise Filter"]) {
            sl::tool::cudaFunc::filterPhase(leftSovleData.__unwrapMap,
                                            filterLeftPhase, 0.5, 5);
            sl::tool::cudaFunc::filterPhase(rightSolveData.__unwrapMap,
                                            filterRightPhase, 0.5, 5);
            __restructor->restruction(filterLeftPhase, filterRightPhase, param,
                                      depthMap);
        } else {
            __restructor->restruction(leftSovleData.__unwrapMap,
                                      rightSolveData.__unwrapMap, param,
                                      depthMap);
        }

        depthMap.download(frameData.__depthMap);
    } else {
        PhaseSolverGroupDataHost leftSovleData, rightSolveData;
        __phaseSolver->solve(leftImgsConvert, leftSovleData,
                             __numbericalProperties["Contrast Threshold"]);
        __phaseSolver->solve(rightImgConvert, rightSolveData,
                             __numbericalProperties["Contrast Threshold"]);

        cv::Mat filterLeftPhase, filterRightPhase;
        if (__booleanProperties["Noise Filter"]) {
            filterPhase(leftSovleData.__unwrapMap, filterLeftPhase);
            filterPhase(rightSolveData.__unwrapMap, filterRightPhase);
            __restructor->restruction(filterLeftPhase, filterRightPhase, param,
                                      frameData.__depthMap);
        } else {
            __restructor->restruction(leftSovleData.__unwrapMap,
                                      rightSolveData.__unwrapMap, param,
                                      frameData.__depthMap);
        }
    }

    if (!colorShiftImgs.empty()) {
        averageTexture(colorShiftImgs, frameData.__textureMap,
            __numbericalProperties["Phase Shift Times"]);
    }
    else {
        averageTexture(leftImgs, frameData.__textureMap,
            __numbericalProperties["Phase Shift Times"]);
        cv::cvtColor(frameData.__textureMap, frameData.__textureMap,
            cv::COLOR_GRAY2RGB);
    }

    cv::Mat depthMapRected;
    auto caliInfo = __matrixInfo->getInfo();
    remapToDepthCamera(frameData.__depthMap, frameData.__textureMap,
                       caliInfo.__Q, caliInfo.__M1, caliInfo.__R1,
                       depthMapRected, frameData.__pointCloud);
    frameData.__depthMap = depthMapRected;

    return true;
}

bool BinocularCamera::capture(FrameData &frameData) {
    const CameraFactory::CameraManufactor manufator =
        __stringProperties["2D Camera Manufactor"] == "Huaray"
            ? CameraFactory::Huaray
            : CameraFactory::Halcon;
    auto pLeftCamera = __cameraFactory.getCamera(
        __stringProperties["Left Camera Name"], manufator);
    auto pRightCamera = __cameraFactory.getCamera(
        __stringProperties["Right Camera Name"], manufator);
    auto pProjector =
        __projectorFactory.getProjector(__stringProperties["DLP Evm"]);
    const int imgSizeWaitFor = __numbericalProperties["Phase Shift Times"] +
                               __numbericalProperties["Gray Code Bits"];

    if (__booleanProperties["Enable Depth Camera"]) {
        pProjector->project(false);
        auto start = std::chrono::system_clock::now();
        double timeMaximumWaitFor = 9999999;
        while (pLeftCamera->getImgs().size() < imgSizeWaitFor ||
               pRightCamera->getImgs().size() < imgSizeWaitFor) {
            auto cur = std::chrono::system_clock::now();
            double timeElapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(cur -
                                                                      start)
                    .count() *
                (double)std::chrono::milliseconds::period::num /
                std::chrono::milliseconds::period::den;
            if (timeElapsed > timeMaximumWaitFor) {
                return false;
            }
        }

        std::vector<cv::Mat> leftImgs(imgSizeWaitFor),
            rightImgs(imgSizeWaitFor);
        // TODO@LiuYunhang:未用多线程加速，测试后视情况而定
        std::vector<cv::Mat> colorShiftImgs;
        for (size_t i = 0; i < leftImgs.size(); ++i) {
            cv::Mat leftImg = pLeftCamera->popImg();
            cv::Mat rightImg = pRightCamera->popImg();
            if (leftImg.type() == CV_8UC3) {
                colorShiftImgs.push_back(leftImg);
                cv::cvtColor(leftImg, leftImgs[i], cv::COLOR_RGB2GRAY);
                __rectifier->remapImg(leftImgs[i], leftImgs[i], true);
            } else {
                __rectifier->remapImg(leftImg, leftImgs[i], true);
            }

            __rectifier->remapImg(rightImg, rightImgs[i], false);
        }

        RestructParamater param;
        param.__minDisparity = __numbericalProperties["MinimumDisparity"];
        param.__maxDisparity = __numbericalProperties["MaximumDisparity"];
        param.__isMapToPreDepthAxes = false;
        param.__isMapToColorCamera = false;
        param.__minDepth = __numbericalProperties["MinimumDepth"];
        param.__maxDepth = __numbericalProperties["MaximumDepth"];
        param.__maximumCost = __numbericalProperties["MaximumMatchCost"];
#ifdef __WITH_CUDA__
        param.__block = dim3(__numbericalProperties["Num of block X"],
                             __numbericalProperties["Num of block Y"]);
#endif //!__WITH_CUDA__
        if (__booleanProperties["Gpu Accelerate"]) {
            PhaseSolverGroupDataDevice leftSovleData, rightSolveData;
            __phaseSolver->solve(leftImgs, leftSovleData,
                                 __numbericalProperties["Contrast Threshold"]);
            __phaseSolver->solve(rightImgs, rightSolveData,
                                 __numbericalProperties["Contrast Threshold"]);

            cv::cuda::GpuMat filterLeftPhase, filterRightPhase, depthMap;
            if (__booleanProperties["Noise Filter"]) {
                sl::tool::cudaFunc::filterPhase(leftSovleData.__unwrapMap,
                                                filterLeftPhase, 0.5, 5);
                sl::tool::cudaFunc::filterPhase(rightSolveData.__unwrapMap,
                                                filterRightPhase, 0.5, 5);
                __restructor->restruction(filterLeftPhase, filterRightPhase,
                                          param, depthMap);
            } else {
                __restructor->restruction(leftSovleData.__unwrapMap,
                                          rightSolveData.__unwrapMap, param,
                                          depthMap);
            }

            depthMap.download(frameData.__depthMap);
        } else {
            PhaseSolverGroupDataHost leftSovleData, rightSolveData;
            __phaseSolver->solve(leftImgs, leftSovleData,
                                 __numbericalProperties["Contrast Threshold"]);
            __phaseSolver->solve(rightImgs, rightSolveData,
                                 __numbericalProperties["Contrast Threshold"]);

            cv::Mat filterLeftPhase, filterRightPhase;
            if (__booleanProperties["Noise Filter"]) {
                filterPhase(leftSovleData.__unwrapMap, filterLeftPhase);
                filterPhase(rightSolveData.__unwrapMap, filterRightPhase);
                __restructor->restruction(filterLeftPhase, filterRightPhase,
                                          param, frameData.__depthMap);
            } else {
                __restructor->restruction(leftSovleData.__unwrapMap,
                                          rightSolveData.__unwrapMap, param,
                                          frameData.__depthMap);
            }

            if (!colorShiftImgs.empty()) {
                averageTexture(colorShiftImgs, frameData.__textureMap,
                               __numbericalProperties["Phase Shift Times"]);
            } else {
                averageTexture(leftImgs, frameData.__textureMap,
                               __numbericalProperties["Phase Shift Times"]);
                cv::cvtColor(frameData.__textureMap, frameData.__textureMap,
                             cv::COLOR_GRAY2RGB);
            }
        }

        cv::Mat depthMapRected;
        auto caliInfo = __matrixInfo->getInfo();
        remapToDepthCamera(frameData.__depthMap, frameData.__textureMap,
                           caliInfo.__Q, caliInfo.__M1, caliInfo.__R1,
                           depthMapRected, frameData.__pointCloud);
        frameData.__depthMap = depthMapRected;

        return true;
    }

    frameData.__textureMap =
        __cameraFactory
            .getCamera(__stringProperties["Left Camera Name"], manufator)
            ->capture();

    return true;
}

bool BinocularCamera::setDepthCameraEnabled(const bool isEnable) {
    return setBooleanAttribute("Enable Depth Camera", isEnable);
}

bool BinocularCamera::getStringAttribute(const std::string attributeName,
                                         std::string &val) {

    if (!__stringProperties.count(attributeName)) {
        printf("property [%s] is not be supported !\n", attributeName.data());
        return false;
    }

    val = __stringProperties[attributeName];

    return true;
}

bool BinocularCamera::getNumbericalAttribute(const std::string attributeName,
                                             double &val) {
    if (!__numbericalProperties.count(attributeName)) {
        printf("property [%s] is not be supported !\n", attributeName.data());
        return false;
    }
    val = __numbericalProperties[attributeName];

    return true;
}

bool BinocularCamera::getBooleanAttribute(const std::string attributeName,
                                          bool &val) {
    if (!__booleanProperties.count(attributeName)) {
        printf("property [%s] is not be supported !\n", attributeName.data());
        return false;
    }

    val = __booleanProperties[attributeName];

    return true;
}

bool BinocularCamera::setStringAttribute(const std::string attributeName,
                                         const std::string val) {
    __stringProperties[attributeName] = val;

    return true;
}

bool BinocularCamera::setNumberAttribute(const std::string attributeName,
                                         const double val) {



    __numbericalProperties[attributeName] = val;

    if(__isInitial)
        __propertiesChangedSignals[attributeName] = true;

    return true;
}

bool BinocularCamera::setBooleanAttribute(const std::string attributeName,
                                          const bool val) {
    __booleanProperties[attributeName] = val;

    if(__isInitial)
        __propertiesChangedSignals[attributeName] = true;

    return true;
}

bool BinocularCamera::resetCameraConfig() {
    __stringProperties["Camera Name"] = "Binocular Camera";
    __stringProperties["Manufactor"] = "@LiuYunhuang";
    __stringProperties["Email"] = "@1369215984@qq.com";
    __stringProperties["Accuracy"] = "0.2mm @1m";
    __stringProperties["Pixels Of Height"] = "1536";
    __stringProperties["Pixels Of Width"] = "2048";
    __stringProperties["True Width"] = "600mm";
    __stringProperties["True Height"] = "500mm";
    __stringProperties["Intrinsic Path"] = __stringProperties["Intrinsic Path"];
    __stringProperties["Extrinsic Path"] = __stringProperties["Extrinsic Path"];
    __stringProperties["DLP Evm"] = "DLP4710";
    __stringProperties["Left Camera Name"] = "Left";
    __stringProperties["RightCamera Name"] = "Right";
    __stringProperties["2D Camera Manufactor"] = "Huaray";

    __numbericalProperties["MinimumDisparity"] = -500;
    __numbericalProperties["MaximumDisparity"] = 500;
    __numbericalProperties["MinimumDepth"] = 500;
    __numbericalProperties["MaximumDepth"] = 1500;
    __numbericalProperties["Contrast Threshold"] = 5;
    __numbericalProperties["Light Strength"] = 0.9;
    __numbericalProperties["Exposure Time"] = 20000;
    __numbericalProperties["Pre Exposure Time"] = 523;
    __numbericalProperties["Aft Exposure Time"] = 93;
    __numbericalProperties["Phase Shift Times"] = 4;
    __numbericalProperties["Gray Code Bits"] = 6;
    __numbericalProperties["MaximumMatchCost"] = 0.1;
    __numbericalProperties["Num of block X"] = 32;
    __numbericalProperties["Num of block Y"] = 8;

    __booleanProperties["Enable Depth Camera"] = true;
    __booleanProperties["Noise Filter"] = true;
    __booleanProperties["Gpu Accelerate"] = false;
    __booleanProperties["Is Vertical"] = true;

    saveParams(__jsonPath, __jsonVal);
    loadParams(__jsonPath, __jsonVal);

    return updateCamera();
}

void BinocularCamera::updateFlashPattern() {
    auto pProjector =
        __projectorFactory.getProjector(__stringProperties["DLP Evm"]);
    auto projectorInfo = pProjector->getInfo();
    cv::Size imgSize = cv::Size(projectorInfo.__width, projectorInfo.__height);

    std::vector<cv::Mat> imgs;

    PhaseEncoder *phaseCoder = new PhaseEncoder(imgSize);
    phaseCoder->setBinaryPhaseCode(false);
    phaseCoder->setOPWMCode(false);
    phaseCoder->setCounterMode(false);
    phaseCoder->setWhiteMode(false);
    phaseCoder->setPhaseErrorExpandMode(false);
    phaseCoder->setCounterDirection(false);
    phaseCoder->setIsOneHeightMode(true);
    phaseCoder->setChangeModeY(!__booleanProperties["Is Vertical"]);
    const float pathDistance =
        __booleanProperties["Is Vertical"]
            ? imgSize.width /
                  std::pow(2.f, __numbericalProperties["Gray Code Bits"] - 1)
            : imgSize.height /
                  std::pow(2.f, __numbericalProperties["Gray Code Bits"] - 1);
    auto phaseImgs = phaseCoder->creatPhaseImage(
        __numbericalProperties["Phase Shift Times"], pathDistance);
    imgs.insert(imgs.end(), phaseImgs.begin(), phaseImgs.end());

    GrayEncoder *grayCoder = new GrayEncoder(imgSize);
    grayCoder->setShiftGrayCodeState(false);
    grayCoder->setCounterFirstBit(false);
    grayCoder->setFourFloorMode(false);
    grayCoder->setIsOneHeightMode(true);
    grayCoder->setChangeModeY(!__booleanProperties["Is Vertical"]);
    for (size_t i = 1; i <= __numbericalProperties["Gray Code Bits"]; ++i) {
        imgs.push_back(grayCoder->creatGrayImage(
            i, 0, __numbericalProperties["Gray Code Bits"]));
    }

    std::vector<PatternOrderSet> orderSets;
    for (size_t i = 0; i < imgs.size(); i = i + 7) {
        PatternOrderSet patterns;
        patterns.__exposureTime = __numbericalProperties["Exposure Time"];
        patterns.__preExposureTime =
            __numbericalProperties["Pre Exposure Time"];
        patterns.__postExposureTime =
            __numbericalProperties["Aft Exposure Time"];
        patterns.__illumination = Illumination::RGB;
        patterns.__isOneBit = false;
        patterns.__isVertical = __booleanProperties["Is Vertical"];
        patterns.__patternArrayCounts =
            __booleanProperties["Is Vertical"] ? imgSize.width : imgSize.height;
        patterns.__invertPatterns = false;
        patterns.__imgs =
            i + 7 < imgs.size()
                ? std::vector<cv::Mat>(imgs.begin() + i, imgs.begin() + i + 7)
                : std::vector<cv::Mat>(imgs.begin() + i, imgs.end());
        orderSets.push_back(patterns);
    }

    pProjector->populatePatternTableData(orderSets);

    delete phaseCoder;
    delete grayCoder;
}

void BinocularCamera::updateExposureTime() {
    const CameraFactory::CameraManufactor manufator =
        __stringProperties["2D Camera Manufactor"] == "Huaray"
            ? CameraFactory::Huaray
            : CameraFactory::Halcon;
    __cameraFactory
        .getCamera(__stringProperties["Left Camera Name"], manufator)
        ->setNumberAttribute("ExposureTime",
                             __numbericalProperties["Exposure Time"]);
    __cameraFactory
        .getCamera(__stringProperties["Right Camera Name"], manufator)
        ->setNumberAttribute("ExposureTime",
                             __numbericalProperties["Exposure Time"]);
}

void BinocularCamera::updateEnableDepthCamera() {
    const CameraFactory::CameraManufactor manufator =
        __stringProperties["2D Camera Manufactor"] == "Huaray"
            ? CameraFactory::Huaray
            : CameraFactory::Halcon;
    __cameraFactory
        .getCamera(__stringProperties["Left Camera Name"], manufator)
        ->setTrigMode(camera::trigLine);
    __cameraFactory
        .getCamera(__stringProperties["Right Camera Name"], manufator)
        ->setTrigMode(trigLine);
}

void BinocularCamera::updateLightStrength() {
    __projectorFactory.getProjector(__stringProperties["DLP Evm"])
        ->setLEDCurrent(__numbericalProperties["Light Strength"],
                        __numbericalProperties["Light Strength"],
                        __numbericalProperties["Light Strength"]);
}

bool BinocularCamera::updateCamera() {
    if (__propertiesChangedSignals["Gpu Accelerate"]) {
        updateAccelerateMethod();
    }

    if (__propertiesChangedSignals["Enable Depth Camera"]) {
        if (__booleanProperties["Enable Depth Camera"]) {
            updateEnableDepthCamera();
        }
    }

    if (__propertiesChangedSignals["Phase Shift Times"] ||
        __propertiesChangedSignals["Gray Code Bits"] ||
        __propertiesChangedSignals["Is Vertical"] ||
        __propertiesChangedSignals["Exposure Time"]) {
        updateExposureTime();
        updateFlashPattern();
    }

    if (__propertiesChangedSignals["Light Strength"]) {
        updateLightStrength();
    }

    saveParams(__jsonPath, __jsonVal);

    for (auto propertyPair : __propertiesChangedSignals) {
        propertyPair.second = false;
    }

    return true;
}
} // namespace slCamera
} // namespace sl