#include <gtest/gtest.h>

#include <projectorFactory.h>

const std::string testProjector4710 = "DLP4710";
const std::string testProjector3010 = "DLP3010";
const std::string testData4710 = "../../test/data/4_4710";
const std::string testData3010 = "../../test/data/4_3010";
/*
TEST(Projector, init) {
    auto projectorFactory = sl::projector::ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    ASSERT_NE(projectorDlpcApi, nullptr);
}

TEST(Project, connect) {
    auto projectorFactory = sl::projector::ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    bool isSucess = projectorDlpcApi->connect();
    ASSERT_EQ(isSucess, true);
    isSucess = projectorDlpcApi->disConnect();
}

TEST(Project, isConnect) {
    auto projectorFactory = sl::projector::ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    bool isSucess = projectorDlpcApi->connect();
    isSucess = projectorDlpcApi->isConnect();
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->stop();
    projectorDlpcApi->disConnect();
}

TEST(Project, disconnect) {
    auto projectorFactory = sl::projector::ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    bool isSucess = projectorDlpcApi->connect();
    ASSERT_EQ(projectorDlpcApi->disConnect(), true);
    projectorDlpcApi->disConnect();
}

TEST(Project, onceProject) {
    auto projectorFactory = sl::projector::ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    bool isSucess = projectorDlpcApi->connect();
    isSucess = projectorDlpcApi->project(false);
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->stop();
    projectorDlpcApi->disConnect();
}

TEST(Project, continueProject) {
    auto projectorFactory = sl::projector::ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    bool isSucess = projectorDlpcApi->connect();
    isSucess = projectorDlpcApi->project(true);
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->stop();
    isSucess = projectorDlpcApi->disConnect();
}

TEST(Project, pause) {
    auto projectorFactory = sl::projector::ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    bool isSucess = projectorDlpcApi->connect();
    isSucess = projectorDlpcApi->project(true);
    isSucess = projectorDlpcApi->pause();
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->stop();
    projectorDlpcApi->disConnect();
}

TEST(Project, resume) {
    auto projectorFactory = sl::projector::ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    bool isSucess = projectorDlpcApi->connect();
    isSucess = projectorDlpcApi->project(true);
    isSucess = projectorDlpcApi->pause();
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    isSucess = projectorDlpcApi->resume();
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->stop();
    projectorDlpcApi->disConnect();
}

TEST(Project, stop) {
    auto projectorFactory = sl::projector::ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    bool isSucess = projectorDlpcApi->connect();
    isSucess = projectorDlpcApi->project(true);
    isSucess = projectorDlpcApi->stop();
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->disConnect();
}

TEST(Project, step) {
    auto projectorFactory = sl::projector::ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    bool isSucess = projectorDlpcApi->connect();
    isSucess = projectorDlpcApi->project(true);
    isSucess = projectorDlpcApi->step();
    isSucess = projectorDlpcApi->step();
    isSucess = projectorDlpcApi->step();
    isSucess = projectorDlpcApi->step();
    isSucess = projectorDlpcApi->stop();
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->disConnect();
}

TEST(Project, getSetLEDCurrent) {
    auto projectorFactory = sl::projector::ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    bool isSucess = projectorDlpcApi->connect();
    double red, green, blue;
    isSucess = projectorDlpcApi->getLEDCurrent(red, green, blue);
    printf("projector's current light strength: red %d, green %d, blue %d", red, green, blue);
    ASSERT_EQ(isSucess, true);
    isSucess = projectorDlpcApi->setLEDCurrent(0.95, 0.95, 0.95);
    ASSERT_EQ(isSucess, true);
    isSucess = projectorDlpcApi->getLEDCurrent(red, green, blue);
    printf("after set light stength, projector's current light strength: red %d, green %d, blue %d", red, green, blue);
    projectorDlpcApi->disConnect();
}

TEST(Projector, getInfo) {
    auto projectorFactory = sl::projector::ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    auto info = projectorDlpcApi->getInfo();
    ASSERT_EQ(info.__isFind, true);
}
*/
TEST(Projector, populatePatternTableData) {
    auto projectorFactory = sl::projector::ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    bool isSucess = projectorDlpcApi->connect();

    std::vector<sl::projector::PatternOrderSet> patternSets(2);
    patternSets[0].__exposureTime = 10000;
    patternSets[0].__preExposureTime = 7000;
    patternSets[0].__postExposureTime = 3000;
    patternSets[0].__illumination = sl::projector::Blue;
    patternSets[0].__invertPatterns = false;
    patternSets[0].__isVertical = true;
    patternSets[0].__isOneBit = false;
    patternSets[0].__patternArrayCounts = 1920;
    patternSets[1].__exposureTime = 10000;
    patternSets[1].__preExposureTime = 7000;
    patternSets[1].__postExposureTime = 3000;
    patternSets[1].__illumination = sl::projector::Blue;
    patternSets[1].__invertPatterns = false;
    patternSets[1].__isVertical = true;
    patternSets[1].__isOneBit = false;
    patternSets[1].__patternArrayCounts = 1920;

    std::vector<cv::String> imgsPaths;
    cv::glob(testData4710, imgsPaths);
    std::vector<cv::Mat> imgFirstSet;
    std::vector<cv::Mat> imgSecondSet;
    for (size_t i = 0; i < imgsPaths.size(); ++i) {
        imgFirstSet.push_back(cv::imread(testData4710 + "/" + std::to_string(i) + ".bmp", 0));
        imgSecondSet.push_back(cv::imread(testData4710 + "/" + std::to_string(i) + ".bmp", 0));
    }
    patternSets[0].__imgs = imgFirstSet;
    patternSets[1].__imgs = imgSecondSet;

    isSucess = projectorDlpcApi->populatePatternTableData(patternSets);
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->project(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    projectorDlpcApi->stop();
    projectorDlpcApi->disConnect();
}