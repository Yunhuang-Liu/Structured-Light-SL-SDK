#include <nShiftLineNGrayCodeMasterCpu.h>

#include <immintrin.h>

namespace sl {
namespace phaseSolver {
void NShiftLineNGrayCodeMasterCpu::solve(const std::vector<cv::Mat> &imgs,
                                         PhaseSolverGroupDataHost &groupData,
                                         const float sncThreshold,
                                         const int grayCodeBits) {
    std::vector<cv::Mat> floatImgs(imgs.size());
    if (imgs[0].type() != CV_32FC1) {
        for (size_t i = 0; i < imgs.size(); ++i) {
            imgs[i].convertTo(floatImgs[i], CV_32FC1);
        }
    }

    const std::vector<cv::Mat> &applyImgs =
        imgs[0].type() != CV_32FC1 ? floatImgs : imgs;

    groupData.__textureMap =
        (applyImgs[applyImgs.size() - 2] + applyImgs[applyImgs.size() - 1]) /
        2.f;

    cv::Mat shiftLineCodeImg =
        cv::Mat(applyImgs[0].size(), CV_32FC1, cv::Scalar(0.f));
    cv::Mat edgeSubPixelImg =
        cv::Mat(applyImgs[0].size(), CV_32FC1, cv::Scalar(0.f));
    int allocateCode = 0;
    for (size_t i = grayCodeBits; i < applyImgs.size() - 2; i = i + 2) {
        cv::Mat imgEdge, imgShiftLineCode;
        locateSubpixelEdge(applyImgs[i], applyImgs[i + 1], allocateCode++,
            imgShiftLineCode, imgEdge);
        shiftLineCodeImg += imgShiftLineCode;
        edgeSubPixelImg += imgEdge;
    }

    cv::Mat grayCodeImg;
    decodeGrayCode(applyImgs, grayCodeBits, groupData.__textureMap, shiftLineCodeImg, grayCodeImg);

    cv::Mat shiftLineMaskCode;
    decodeShiftLineMaskCode(applyImgs, grayCodeBits, groupData.__textureMap, shiftLineCodeImg,
                            shiftLineMaskCode);

    std::vector<cv::Mat> chanelsPartImgs = {edgeSubPixelImg, shiftLineMaskCode,
                                            shiftLineCodeImg, grayCodeImg};
    cv::merge(chanelsPartImgs, groupData.__unwrapMap);

    groupData.__wrapMap = shiftLineCodeImg;
}

void NShiftLineNGrayCodeMasterCpu::locateSubpixelEdge(
    const cv::Mat &positiveImg, const cv::Mat &negtiveImg,
    const int allocateEdgeCode, cv::Mat &shiftLineCodeImg, cv::Mat &edgeImg) {
    cv::Mat diffImg = positiveImg - negtiveImg;
    shiftLineCodeImg =
        cv::Mat(positiveImg.rows, positiveImg.cols, CV_32FC1, cv::Scalar(0));
    edgeImg =
        cv::Mat(positiveImg.rows, positiveImg.cols, CV_32FC1, cv::Scalar(0));

    const int rows = positiveImg.rows;
    const int cols = positiveImg.cols;
    std::vector<std::thread> tasks(std::thread::hardware_concurrency());
    int rowsPerThread = rows / tasks.size();
    for (int i = 0; i < tasks.size() - 1; i++) {
        tasks[i] =
            std::thread(&NShiftLineNGrayCodeMasterCpu::entryLocateSubpixelEdge,
                        this, std::ref(diffImg), std::ref(positiveImg),
                        std::ref(negtiveImg), allocateEdgeCode,
                        cv::Point2i(rowsPerThread * i, rowsPerThread * (i + 1)),
                        std::ref(shiftLineCodeImg), std::ref(edgeImg));
    }

    tasks[tasks.size() - 1] = std::thread(
        &NShiftLineNGrayCodeMasterCpu::entryLocateSubpixelEdge, this,
        std::ref(diffImg), std::ref(positiveImg), std::ref(negtiveImg),
        allocateEdgeCode, cv::Point2i(rowsPerThread * (tasks.size() - 1), rows),
        std::ref(shiftLineCodeImg), std::ref(edgeImg));

    for (int i = 0; i < tasks.size(); i++) {
        tasks[i].join();
    }
}

void NShiftLineNGrayCodeMasterCpu::entryLocateSubpixelEdge(
    const cv::Mat &diffImg, const cv::Mat &positiveImg,
    const cv::Mat &negtiveImg, const int allocateEdgeCode,
    const cv::Point2i region, cv::Mat &shiftLineCodeImg, cv::Mat &edgeImg) {
    for (size_t i = region.x; i < region.y; ++i) {
        auto ptrDiffImg = diffImg.ptr<float>(i);
        auto ptrPositiveImg = positiveImg.ptr<float>(i);
        auto ptrNegtiveImg = negtiveImg.ptr<float>(i);
        auto ptrShiftLineCodeImg = shiftLineCodeImg.ptr<float>(i);
        auto ptrEdgeImg = edgeImg.ptr<float>(i);
        for (size_t j = 0; j < diffImg.cols - 1; ++j) {
            if ((ptrDiffImg[j] > 0) ^ (ptrDiffImg[j + 1] > 0)) {
                if (ptrPositiveImg[j] != ptrPositiveImg[j + 1] &&
                    ptrNegtiveImg[j] != ptrNegtiveImg[j + 1]) {
                    cv::Vec4f linePositive, lineNegtive;
                    std::vector<cv::Point2f> pointsPositive = {
                        cv::Point2f(j, ptrPositiveImg[j]),
                        cv::Point2f(j + 1, ptrPositiveImg[j + 1])};
                    std::vector<cv::Point2f> pointsNegtive = {
                        cv::Point2f(j, ptrNegtiveImg[j]),
                        cv::Point2f(j + 1, ptrNegtiveImg[j + 1])};
                    cv::fitLine(pointsPositive, linePositive, cv::DIST_L2, 0.01,
                                0.01, 0.01);
                    cv::fitLine(pointsNegtive, lineNegtive, cv::DIST_L2, 0.01,
                                0.01, 0.01);

                    cv::Point2f acrossPoint =
                        getCrossPoint(linePositive, lineNegtive);
                    ptrShiftLineCodeImg[j] = allocateEdgeCode;
                    ptrEdgeImg[j] = (acrossPoint.x == -1 || std::abs(acrossPoint.x - j) > 1) ? j : acrossPoint.x;
                } else {
                    ptrShiftLineCodeImg[j] = allocateEdgeCode;
                    ptrEdgeImg[j] = j;
                }
            }
        }
    }
}

cv::Point2f NShiftLineNGrayCodeMasterCpu::getCrossPoint(cv::Vec4f lineA,
                                                        cv::Vec4f lineB) {
    float a1 = -lineA[1];
    float b1 = lineA[0];
    float c1 = lineA[0] * lineA[3] - lineA[1] * lineA[2];

    float a2 = -lineB[1];
    float b2 = lineB[0];
    float c2 = lineB[0] * lineB[3] - lineB[1] * lineB[2];

    float Det = a1 * b2 - a2 * b1;
    float Det1 = c1 * b2 - c2 * b1;
    float Det2 = a1 * c2 - a2 * c1;

    cv::Point2f pt;
    if (std::abs(Det) < 0.001) {
        pt = {-1, -1};
        return pt;
    }
    pt = {Det1 / Det, Det2 / Det};

    return pt;
}

void NShiftLineNGrayCodeMasterCpu::decodeGrayCode(
    const std::vector<cv::Mat> &imgs, const int grayCodeBits,
    const cv::Mat &conditionImg, const cv::Mat &shiftCodeImg, cv::Mat &grayCodeImg) {

    grayCodeImg =
        cv::Mat(conditionImg.rows, conditionImg.cols, CV_32FC1, cv::Scalar(0));

    const int rows = grayCodeImg.rows;
    const int cols = grayCodeImg.cols;
    std::vector<std::thread> tasks(std::thread::hardware_concurrency());
    int rowsPerThread = rows / tasks.size();
    for (int i = 0; i < tasks.size() - 1; i++) {
        tasks[i] = std::thread(
            &NShiftLineNGrayCodeMasterCpu::entryDecodeGrayCode, this,
            std::ref(imgs), grayCodeBits, std::ref(conditionImg),
            std::ref(shiftCodeImg),
            cv::Point2i(rowsPerThread * i, rowsPerThread * (i + 1)),
            std::ref(grayCodeImg));
    }

    tasks[tasks.size() - 1] =
        std::thread(&NShiftLineNGrayCodeMasterCpu::entryDecodeGrayCode, this,
                    std::ref(imgs), grayCodeBits, std::ref(conditionImg),
                    std::ref(shiftCodeImg),
                    cv::Point2i(rowsPerThread * (tasks.size() - 1), rows),
                    std::ref(grayCodeImg));

    for (int i = 0; i < tasks.size(); i++) {
        tasks[i].join();
    }
}

void NShiftLineNGrayCodeMasterCpu::entryDecodeGrayCode(
    const std::vector<cv::Mat> &imgs, const int grayCodeBits,
    const cv::Mat &conditionImg, const cv::Mat &shiftCodeImg, const cv::Point2i region,
    cv::Mat &grayCodeImg) {
    const int rows = grayCodeImg.rows;
    const int cols = grayCodeImg.cols;
    __m256 zero = _mm256_set1_ps(0);
    __m256 one = _mm256_set1_ps(1);
    __m256 two = _mm256_set1_ps(2);

    for (int i = region.x; i < region.y; i++) {
        std::vector<const float *> ptrImgs(grayCodeBits);
        for (int gray = 0; gray < grayCodeBits; ++gray)
            ptrImgs[gray] = imgs[gray].ptr<float>(i);
        const float *pCondition = conditionImg.ptr<float>(i);
        const float *pShiftCodeImg = shiftCodeImg.ptr<float>(i);
        float *ptr_absoluteImg = grayCodeImg.ptr<float>(i);
        for (int j = 0; j < cols; j += 8) {
            __m256 sumDataK2 = _mm256_set1_ps(0);
            __m256 sumDataK1 = _mm256_set1_ps(0);
            __m256 conditionData = _mm256_load_ps(&pCondition[j]);
            __m256 preBitData = _mm256_set1_ps(0);
            for (int gray = 0; gray < grayCodeBits; ++gray) {
                __m256 imgData = _mm256_load_ps(&ptrImgs[gray][j]);
                __m256 compareData = _mm256_and_ps(
                    _mm256_cmp_ps(imgData, conditionData, _CMP_GE_OS),
                    one);

                __m256 curBitData = _mm256_xor_ps(
                        compareData,
                        preBitData);

                sumDataK2 = _mm256_add_ps(
                    sumDataK2,
                    _mm256_mul_ps(curBitData, _mm256_set1_ps(pow(2, grayCodeBits - 1 - gray))));

                if(gray != grayCodeBits - 1) {
                    sumDataK1 = _mm256_add_ps(
                        sumDataK1,
                        _mm256_mul_ps(curBitData, _mm256_set1_ps(pow(2, grayCodeBits - 2 - gray))));                    
                }

                preBitData = curBitData;
            }

            __m256 shiftCodeData = _mm256_load_ps(&pShiftCodeImg[j]);
            __m256 zeroShiftCodeState = _mm256_and_ps(
                    _mm256_cmp_ps(shiftCodeData, zero, _CMP_EQ_OS),
                    one);
            __m256 otherState = _mm256_sub_ps(one, zeroShiftCodeState);
            __m256 sumData = _mm256_add_ps(_mm256_mul_ps(zeroShiftCodeState, _mm256_div_ps(_mm256_add_ps(sumDataK2, one), two)), _mm256_mul_ps(otherState, sumDataK1));
            _mm256_store_ps(&ptr_absoluteImg[j], sumData);
        }
    }
}

void NShiftLineNGrayCodeMasterCpu::decodeShiftLineMaskCode(
    const std::vector<cv::Mat> &imgs, const int grayCodeBits,
    const cv::Mat &conditionImg, const cv::Mat &shiftCodeImg, cv::Mat &maskCodeImg) {

    maskCodeImg =
        cv::Mat(conditionImg.rows, conditionImg.cols, CV_32FC1, cv::Scalar(0));

    const int rows = maskCodeImg.rows;
    const int cols = maskCodeImg.cols;
    std::vector<std::thread> tasks(std::thread::hardware_concurrency());
    int rowsPerThread = rows / tasks.size();
    for (int i = 0; i < tasks.size() - 1; i++) {
        tasks[i] = std::thread(
            &NShiftLineNGrayCodeMasterCpu::entryDecodeShiftLineMaskCode, this,
            std::ref(imgs), grayCodeBits, std::ref(conditionImg),
            std::ref(shiftCodeImg),
            cv::Point2i(rowsPerThread * i, rowsPerThread * (i + 1)),
            std::ref(maskCodeImg));
    }

    tasks[tasks.size() - 1] =
        std::thread(&NShiftLineNGrayCodeMasterCpu::entryDecodeShiftLineMaskCode,
                    this, std::ref(imgs), grayCodeBits, std::ref(conditionImg),
                    std::ref(shiftCodeImg),
                    cv::Point2i(rowsPerThread * (tasks.size() - 1), rows),
                    std::ref(maskCodeImg));

    for (int i = 0; i < tasks.size(); i++) {
        tasks[i].join();
    }
}

void NShiftLineNGrayCodeMasterCpu::entryDecodeShiftLineMaskCode(
    const std::vector<cv::Mat> &imgs, const int grayCodeBits,
    const cv::Mat &conditionImg, const cv::Mat& shiftCodeImg, const cv::Point2i region,
    cv::Mat &maskCodeImg) {
    const int shiftLineNums = (imgs.size() - 2 - grayCodeBits) / 2;

    for (size_t i = region.x; i < region.y; ++i) {
        std::vector<const float *> ptrShiftLines(shiftLineNums);
        int index = 0;
        for (size_t j = grayCodeBits; j < imgs.size() - 2; j += 2) {
            ptrShiftLines[index++] = imgs[j].ptr<float>(i);
        }

        float *ptrMaskCodeImg = maskCodeImg.ptr<float>(i);
        const float *ptrConditionImg = conditionImg.ptr<float>(i);
        const float *ptrShiftCodeImg = shiftCodeImg.ptr<float>(i);

        for (size_t k = 0; k < conditionImg.cols; ++k) {
            int bitNum = 0, maskCode = 0;
            for (size_t d = 0; d < ptrShiftLines.size(); ++d) {
                if(d == ptrShiftCodeImg[k]) {
                    continue;
                }
                maskCode += ((ptrShiftLines[d][k] > ptrConditionImg[k]) * std::pow(2, bitNum++));
            }

            ptrMaskCodeImg[k] = maskCode;
        }
    }
}
} // namespace phaseSolver
} // namespace sl