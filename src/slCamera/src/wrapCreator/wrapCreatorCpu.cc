#include <wrapCreatorCpu.h>

namespace sl {
namespace wrapCreator {
void WrapCreatorCpu::getWrapImg(const std::vector<cv::Mat> &imgs,
                                cv::Mat &wrapImg, cv::Mat &conditionImg) {
    std::vector<cv::Mat> floatImgs(imgs.size());
    if(imgs[0].type() != CV_32FC1) {
        for(size_t i = 0; i < imgs.size(); ++i) {
            imgs[i].convertTo(floatImgs[i], CV_32FC1);
        }
    }

    const std::vector<cv::Mat>& applyImgs = imgs[0].type() != CV_32FC1 ? floatImgs : imgs;
    wrapImg = cv::Mat(imgs[0].size(), CV_32FC1, cv::Scalar(0.f));
    conditionImg = cv::Mat(imgs[0].size(), CV_32FC1, cv::Scalar(0.f));

    std::vector<std::thread> tasks(std::thread::hardware_concurrency());
    int rows = wrapImg.rows / tasks.size();
    for (int i = 0; i < tasks.size() - 1; i++) {
        tasks[i] = std::thread(&WrapCreatorCpu::entryWrap,
                               this, std::ref(applyImgs),
                               cv::Point2i(rows * i, rows * (i + 1)),
                               std::ref(wrapImg));
    }
    tasks[tasks.size() - 1] = std::thread(
        &WrapCreatorCpu::entryWrap, this, std::ref(applyImgs),
        cv::Point2i(rows * (tasks.size() - 1), wrapImg.rows),
        std::ref(wrapImg));
    for (int i = 0; i < tasks.size(); i++) {
        if (tasks[i].joinable()) {
            tasks[i].join();
        }
    }

    solveTextureImgs(applyImgs, conditionImg);
}

void WrapCreatorCpu::solveTextureImgs(const std::vector<cv::Mat> &imgs,
                                      cv::Mat &conditionImg) {
    const int rows = imgs[0].rows;
    const int cols = imgs[0].cols;
    std::vector<std::thread> tasks(std::thread::hardware_concurrency());
    int rowsPerThread = rows / tasks.size();
    for (int i = 0; i < tasks.size() - 1; ++i) {
        tasks[i] =
            std::thread(&WrapCreatorCpu::entrySolveTextureImgs, this,
                        std::ref(imgs),
                        cv::Point2i(rowsPerThread * i, rowsPerThread * (i + 1)),
                        std::ref(conditionImg));
    }

    tasks[tasks.size() - 1] = std::thread(
        &WrapCreatorCpu::entrySolveTextureImgs, this, std::ref(imgs), cv::Point2i(rowsPerThread * (tasks.size() - 1), rows),
        std::ref(conditionImg));

    for (int i = 0; i < tasks.size(); ++i) {
        if (tasks[i].joinable()) {
            tasks[i].join();
        }
    }
}

void WrapCreatorCpu::entrySolveTextureImgs(const std::vector<cv::Mat> &imgs,
                                           const cv::Point2i region,
                                           cv::Mat &conditionImg) {
    __m256 shiftStepData = _mm256_set1_ps(imgs.size());
    std::vector<__m256> dataImgs(imgs.size());
    const int rows = conditionImg.rows;
    const int cols = conditionImg.cols;
    __m256 sumShiftImg = _mm256_set1_ps(0);
    for (int i = region.x; i < region.y; i++) {
        std::vector<const float *> ptrImgs(imgs.size());
        for (int step = 0; step < imgs.size(); ++step)
            ptrImgs[step] = imgs[step].ptr<float>(i);
        float *ptr_conditionImg = conditionImg.ptr<float>(i);
        for (int j = 0; j < cols; j += 8) {
            sumShiftImg = _mm256_set1_ps(0);
            for (int step = 0; step < imgs.size(); ++step) {
                dataImgs[step] = _mm256_load_ps(&ptrImgs[step][j]);
                sumShiftImg = _mm256_add_ps(sumShiftImg, dataImgs[step]);
            }
            _mm256_store_ps(&ptr_conditionImg[j],
                            _mm256_div_ps(sumShiftImg, shiftStepData));
        }
    }
}

void WrapCreatorCpu::entryWrap(const std::vector<cv::Mat> &imgs,
                               const cv::Point2i &region,
                               cv::Mat &wrapImg) {
    __m256 dataCounter1 = _mm256_set1_ps(-1.f);
    __m256 shiftDistance = _mm256_set1_ps(CV_2PI / imgs.size());
    const int cols = wrapImg.cols;
    std::vector<__m256> datas(imgs.size());
    __m256 sinPartial = _mm256_set1_ps(0);
    __m256 cosPartial = _mm256_set1_ps(0);
    __m256 time = _mm256_set1_ps(0);
    __m256 shiftNow = _mm256_set1_ps(0);
    __m256 result = _mm256_set1_ps(0);
    for (size_t i = region.x; i < region.y; i++) {
        std::vector<const float *> ptrImgs(imgs.size());
        for (int step = 0; step < imgs.size(); ++step)
            ptrImgs[step] = imgs[step].ptr<float>(i);
        float *ptr_wrapImg = wrapImg.ptr<float>(i);
        for (size_t j = 0; j < cols; j += 8) {
            for (int step = 0; step < imgs.size(); ++step)
                datas[step] = _mm256_load_ps(&ptrImgs[step][j]);
            sinPartial = _mm256_set1_ps(0);
            cosPartial = _mm256_set1_ps(0);
            for (int step = 0; step < imgs.size(); ++step) {
                time = _mm256_set1_ps(step);
                shiftNow = _mm256_mul_ps(shiftDistance, time);
                sinPartial = _mm256_add_ps(
                    sinPartial,
                    _mm256_mul_ps(datas[step], _mm256_sin_ps(shiftNow)));
                cosPartial = _mm256_add_ps(
                    cosPartial,
                    _mm256_mul_ps(datas[step], _mm256_cos_ps(shiftNow)));
            }
            result = _mm256_mul_ps(dataCounter1,
                                   _mm256_atan2_ps(sinPartial, cosPartial));
            _mm256_store_ps(&ptr_wrapImg[j], result);
        }
    }
}
} // namespace wrapCreator
} // namespace sl