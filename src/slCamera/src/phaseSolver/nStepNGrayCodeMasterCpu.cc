#include <nStepNGrayCodeMasterCpu.h>

#include <immintrin.h>

namespace sl {
namespace phaseSolver {
void NStepNGrayCodeMasterCpu::solveTextureImgs(const std::vector<cv::Mat> &imgs,
                                               cv::Mat &conditionImg,
                                               const int shiftStep) {
    const int rows = imgs[0].rows;
    const int cols = imgs[0].cols;
    std::vector<std::thread> tasks(std::thread::hardware_concurrency());
    int rowsPerThread = rows / tasks.size();
    for (int i = 0; i < tasks.size() - 1; ++i) {
        tasks[i] =
            std::thread(&NStepNGrayCodeMasterCpu::entrySolveTextureImgs, this,
                        std::ref(imgs), shiftStep, 
                        cv::Point2i(rowsPerThread * i, rowsPerThread * (i + 1)),
                        std::ref(conditionImg));
    }

    tasks[tasks.size() - 1] =
        std::thread(&NStepNGrayCodeMasterCpu::entrySolveTextureImgs, this,
                    std::ref(imgs), shiftStep,
                    cv::Point2i(rowsPerThread * (tasks.size() - 1), rows),
                    std::ref(conditionImg));

    for (int i = 0; i < tasks.size(); ++i) {
        if (tasks[i].joinable()) {
            tasks[i].join();
        }
    }
}

void NStepNGrayCodeMasterCpu::entrySolveTextureImgs(
    const std::vector<cv::Mat> &imgs, const int shiftStep, const cv::Point2i region, cv::Mat &conditionImg) {
    __m256 shiftStepData = _mm256_set1_ps(shiftStep);
    const int rows = conditionImg.rows;
    const int cols = conditionImg.cols;

    for (int i = region.x; i < region.y; i++) {
        std::vector<const float *> ptrImgs(shiftStep);
        for (int step = 0; step < shiftStep; ++step)
            ptrImgs[step] = imgs[step].ptr<float>(i);
        float *ptr_conditionImg = conditionImg.ptr<float>(i);
        for (int j = 0; j < cols; j += 8) {
            __m256 sumShiftImg = _mm256_set1_ps(0);
            for (int step = 0; step < shiftStep; ++step) {
                __m256 dataImg = _mm256_load_ps(&ptrImgs[step][j]);
                sumShiftImg = _mm256_add_ps(sumShiftImg, dataImg);
            }
            _mm256_store_ps(&ptr_conditionImg[j], _mm256_div_ps(sumShiftImg, shiftStepData));
        }
    }
}

void NStepNGrayCodeMasterCpu::entryUnwrap(const std::vector<cv::Mat> &imgs,
                                          const int shiftStep,
                                          const cv::Mat &conditionImg,
                                          const float sncThreshold,
                                          const cv::Mat &wrapImg,
                                          const cv::Point2i region,
                                          cv::Mat &absolutePhaseImg) {
    const int rows = absolutePhaseImg.rows;
    const int cols = absolutePhaseImg.cols;
    __m256 add_1_ = _mm256_set1_ps(1);
    __m256 div_2_ = _mm256_set1_ps(2);
    __m256 conditionThreshold = _mm256_set1_ps(sncThreshold);
    __m256 _Counter_PI_Div_2_ = _mm256_set1_ps(-CV_PI / 2);
    __m256 _PI_Div_2_ = _mm256_set1_ps(CV_PI / 2);
    __m256 _2PI_ = _mm256_set1_ps(CV_2PI);
    __m256 zero = _mm256_set1_ps(0);
    __m256 one = _mm256_set1_ps(1);

    const int numOfGrayCode = imgs.size() - shiftStep;
    for (int i = region.x; i < region.y; i++) {
        std::vector<const float *> ptrImgs(numOfGrayCode);
        for (int gray = shiftStep; gray < imgs.size(); ++gray)
            ptrImgs[gray - shiftStep] = imgs[gray].ptr<float>(i);
        const float *pWrapImg = wrapImg.ptr<float>(i);
        const float *pCondition = conditionImg.ptr<float>(i);
        float *ptr_absoluteImg = absolutePhaseImg.ptr<float>(i);
        for (int j = 0; j < cols; j += 8) {
            __m256 sumDataK2 = _mm256_set1_ps(0);
            __m256 sumDataK1 = _mm256_set1_ps(0);
            __m256 preBitData = _mm256_set1_ps(0);
            __m256 wrapImgData = _mm256_load_ps(&pWrapImg[j]);
            __m256 conditionData = _mm256_load_ps(&pCondition[j]);

            for (int gray = 0; gray < numOfGrayCode; ++gray) {
                __m256 imgData = _mm256_load_ps(&ptrImgs[gray][j]);
                __m256 compareData = _mm256_and_ps(
                    _mm256_cmp_ps(imgData, conditionData, _CMP_GE_OS),
                    one);

                __m256 curBitData = _mm256_xor_ps(
                        compareData,
                        preBitData);

                sumDataK2 = _mm256_add_ps(
                    sumDataK2,
                    _mm256_mul_ps(curBitData, _mm256_set1_ps(pow(2, numOfGrayCode - 1 - gray))));

                if(gray != numOfGrayCode - 1) {
                    sumDataK1 = _mm256_add_ps(
                        sumDataK1,
                        _mm256_mul_ps(curBitData, _mm256_set1_ps(pow(2, numOfGrayCode - 2 - gray))));                    
                }

                preBitData = curBitData;
            }
            __m256 K2 = _mm256_floor_ps(
                _mm256_div_ps(_mm256_add_ps(sumDataK2, add_1_), div_2_));
            __m256 K1 = sumDataK1;
            __m256 lessEqualThan = _mm256_and_ps(
                _mm256_cmp_ps(wrapImgData, _Counter_PI_Div_2_, _CMP_LE_OS),
                one);
            __m256 greaterEqualThan = _mm256_and_ps(
                _mm256_cmp_ps(wrapImgData, _PI_Div_2_, _CMP_GE_OS), one);
            __m256 less_data_greaterThan = _mm256_xor_ps(
                _mm256_or_ps(lessEqualThan, greaterEqualThan), one);
            __m256 data_1_ = _mm256_mul_ps(
                lessEqualThan, _mm256_fmadd_ps(_2PI_, K2, wrapImgData));
            __m256 data_2_ = _mm256_mul_ps(
                greaterEqualThan,
                _mm256_fmadd_ps(_2PI_, _mm256_sub_ps(K2, one), wrapImgData));
            __m256 data = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(less_data_greaterThan,
                                  _mm256_fmadd_ps(_2PI_, K1, wrapImgData)),
                    data_1_),
                data_2_);

            __m256 compareCondition =
                _mm256_cmp_ps(conditionData, conditionThreshold, _CMP_GT_OS);
            _mm256_store_ps(
                &ptr_absoluteImg[j],
                _mm256_mul_ps(data, _mm256_and_ps(compareCondition, one)));
        }
    }
}

void NStepNGrayCodeMasterCpu::solve(const std::vector<cv::Mat> &imgs,
                                    PhaseSolverGroupDataHost &groupData,
                                    const float sncThreshold,
                                    const int shiftTime) {
    std::vector<cv::Mat> floatImgs(imgs.size());
    if(imgs[0].type() != CV_32FC1) {
        for(size_t i = 0; i < imgs.size(); ++i) {
            imgs[i].convertTo(floatImgs[i], CV_32FC1);
        }
    }

    const std::vector<cv::Mat>& applyImgs = imgs[0].type() != CV_32FC1 ? floatImgs : imgs;

    groupData.__unwrapMap = cv::Mat(applyImgs[0].size(), CV_32FC1, cv::Scalar(0));
    groupData.__textureMap = cv::Mat(applyImgs[0].size(), CV_32FC1, cv::Scalar(0));
    groupData.__wrapMap = cv::Mat(applyImgs[0].size(), CV_32FC1, cv::Scalar(0));
    getPrepareDate(applyImgs, groupData.__wrapMap, groupData.__textureMap, shiftTime);
    const int rows = groupData.__unwrapMap.rows;
    const int cols = groupData.__unwrapMap.cols;
    std::vector<std::thread> tasks(std::thread::hardware_concurrency());
    int rowsPerThread = rows / tasks.size();
    for (int i = 0; i < tasks.size() - 1; i++) {
        tasks[i] = std::thread(
            &NStepNGrayCodeMasterCpu::entryUnwrap, this, std::ref(applyImgs),
            shiftTime, std::ref(groupData.__textureMap), sncThreshold,
            std::ref(groupData.__wrapMap),
            cv::Point2i(rowsPerThread * i, rowsPerThread * (i + 1)),
            std::ref(groupData.__unwrapMap));
    }

    tasks[tasks.size() - 1] = std::thread(
        &NStepNGrayCodeMasterCpu::entryUnwrap, this, std::ref(applyImgs), shiftTime,
        std::ref(groupData.__textureMap), sncThreshold, std::ref(groupData.__wrapMap),
        cv::Point2i(rowsPerThread * (tasks.size() - 1), rows),
        std::ref(groupData.__unwrapMap));

    for (int i = 0; i < tasks.size(); i++) {
        tasks[i].join();
    }
}

void NStepNGrayCodeMasterCpu::getPrepareDate(const std::vector<cv::Mat> &imgs,
                                             cv::Mat &wrapImg,
                                             cv::Mat &conditionImg,
                                             const int shiftStep) {
    std::vector<std::thread> tasks(std::thread::hardware_concurrency());
    int rows = wrapImg.rows / tasks.size();
    for (int i = 0; i < tasks.size() - 1; i++) {
        tasks[i] = std::thread(&phaseSolver::NStepNGrayCodeMasterCpu::entryWrap,
                               this, std::ref(imgs), shiftStep,
                               cv::Point2i(rows * i, rows * (i + 1)),
                               std::ref(wrapImg));
    }
    tasks[tasks.size() - 1] = std::thread(
        &NStepNGrayCodeMasterCpu::entryWrap, this, std::ref(imgs), shiftStep,
        cv::Point2i(rows * (tasks.size() - 1), wrapImg.rows),
        std::ref(wrapImg));
    for (int i = 0; i < tasks.size(); i++) {
        if (tasks[i].joinable()) {
            tasks[i].join();
        }
    }

    solveTextureImgs(imgs, conditionImg, shiftStep);
}

void NStepNGrayCodeMasterCpu::entryWrap(const std::vector<cv::Mat> &imgs,
                                        const int shiftStep,
                                        const cv::Point2i &region,
                                        cv::Mat &wrapImg) {
    __m256 dataCounter1 = _mm256_set1_ps(-1.f);
    __m256 shiftDistance = _mm256_set1_ps(CV_2PI / shiftStep);
    const int cols = wrapImg.cols;
    __m256 time = _mm256_set1_ps(0);
    __m256 shiftNow = _mm256_set1_ps(0);
    __m256 result = _mm256_set1_ps(0);
    for (size_t i = region.x; i < region.y; i++) {
        std::vector<const float *> ptrImgs(shiftStep);
        for (int step = 0; step < shiftStep; ++step)
            ptrImgs[step] = imgs[step].ptr<float>(i);
        float *ptr_wrapImg = wrapImg.ptr<float>(i);
        for (size_t j = 0; j < cols; j += 8) {
            __m256 sinPartial = _mm256_set1_ps(0);
            __m256 cosPartial = _mm256_set1_ps(0);
            for (int step = 0; step < shiftStep; ++step) {
                time = _mm256_set1_ps(step);
                shiftNow = _mm256_mul_ps(shiftDistance, time);

                __m256 shiftSinVal = _mm256_set1_ps(0);
                __m256 shiftCosVal = _mm256_set1_ps(0);
                for (size_t k = 0; k < 8; ++k) {
                    shiftSinVal[k] = sin(shiftNow[k]);
                    shiftCosVal[k] = cos(shiftNow[k]);
                }

                __m256 dataImg = _mm256_load_ps(&ptrImgs[step][j]);

                sinPartial = _mm256_add_ps(
                    sinPartial,
                    _mm256_mul_ps(dataImg, shiftSinVal));
                cosPartial = _mm256_add_ps(
                    cosPartial,
                    _mm256_mul_ps(dataImg, shiftCosVal));
            }

            __m256 shiftAtanVal = _mm256_set1_ps(0);
            for (size_t k = 0; k < 8; ++k) {
                shiftAtanVal[k] = std::atan2(sinPartial[k], cosPartial[k]);
            }

            result = _mm256_mul_ps(dataCounter1, shiftAtanVal);
            _mm256_store_ps(&ptr_wrapImg[j], result);
        }
    }
}
} // namespace phaseSolver
} // namespace sl