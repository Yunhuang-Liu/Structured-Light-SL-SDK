#include <tool.h>

namespace sl {
namespace tool {
void phaseHeightMapEigCoe(const cv::Mat &phase, const cv::Mat &intrinsic,
                          const cv::Mat &coefficient, const float minDepth,
                          const float maxDepth, cv::Mat &depth) {
    CV_Assert(phase.type() == CV_32FC1);
    CV_Assert(intrinsic.type() == CV_64FC1);
    CV_Assert(coefficient.type() == CV_64FC1);
    depth = cv::Mat::zeros(phase.size(), CV_32FC1);

    const int threads = std::thread::hardware_concurrency();

    const int rows = phase.rows;
    const int cols = phase.cols;
    const int patchRow = rows / threads;

    std::vector<std::thread> threadsPool(threads);
    for (int i = 0; i < threads - 1; ++i) {
        threadsPool[i] = std::thread(
            &phaseHeightMapEigCoeRegion, std::ref(phase), std::ref(intrinsic),
            std::ref(coefficient), minDepth, maxDepth, patchRow * i,
            patchRow * (i + 1), std::ref(depth));
    }
    threadsPool[threads - 1] = std::thread(
        &phaseHeightMapEigCoeRegion, std::ref(phase), std::ref(intrinsic),
        std::ref(coefficient), minDepth, maxDepth, patchRow * (threads - 1),
        phase.rows, std::ref(depth));
    for (auto &thread : threadsPool)
        thread.join();
}

void phaseHeightMapEigCoeRegion(const cv::Mat &phase, const cv::Mat &intrinsic,
                                const cv::Mat &coefficient,
                                const float minDepth, const float maxDepth,
                                const int rowBegin, const int rowEnd,
                                cv::Mat &depth) {
    CV_Assert(intrinsic.type() == CV_64FC1);
    CV_Assert(coefficient.type() == CV_64FC1);
    CV_Assert(depth.type() == CV_32FC1);
    CV_Assert(!depth.empty());

    cv::Mat intrinsicFT;
    intrinsic.convertTo(intrinsicFT, CV_32FC1);

    Eigen::Matrix3f mapL;
    Eigen::Vector3f mapR, cameraPoint, imgPoint;
    mapL << intrinsicFT.ptr<float>(0)[0], 0, 0, 0, intrinsicFT.ptr<float>(1)[1],
        0, 0, 0, 0;
    mapR << 0, 0, 0;

    for (int i = rowBegin; i < rowEnd; ++i) {
        const float *ptrPhase = phase.ptr<float>(i);
        float *ptrDepth = depth.ptr<float>(i);
        for (int j = 0; j < phase.cols; ++j) {
            if (ptrPhase[j] == -5.f) {
                ptrDepth[j] = 0.f;
                continue;
            }
            mapL(0, 2) = intrinsic.ptr<double>(0)[2] - j;
            mapL(1, 2) = intrinsic.ptr<double>(1)[2] - i;
            mapL(2, 0) = coefficient.ptr<double>(0)[0] -
                         coefficient.ptr<double>(4)[0] * ptrPhase[j];
            mapL(2, 1) = coefficient.ptr<double>(1)[0] -
                         coefficient.ptr<double>(5)[0] * ptrPhase[j];
            mapL(2, 2) = coefficient.ptr<double>(2)[0] -
                         coefficient.ptr<double>(6)[0] * ptrPhase[j];

            mapR(2, 0) = coefficient.ptr<double>(7)[0] * ptrPhase[j] -
                         coefficient.ptr<double>(3)[0];
            cameraPoint = mapL.inverse() * mapR;

            ptrDepth[j] = cameraPoint.z();
        }
    }
}

void averageTexture(const std::vector<cv::Mat> &imgs, cv::Mat &texture,
                    const int phaseShiftStep) {
    CV_Assert(imgs.size() >= phaseShiftStep);
    CV_Assert(imgs[0].type() == CV_8UC1 || imgs[0].type() == CV_8UC3);

    const int threads = std::thread::hardware_concurrency();

    const bool isColor = imgs[0].type() == CV_8UC3;
    if (isColor)
        texture = cv::Mat(imgs[0].size(), imgs[0].type(), cv::Scalar(0, 0, 0));
    else
        texture = cv::Mat(imgs[0].size(), imgs[0].type(), cv::Scalar(0));

    const int perRow = texture.rows / threads;
    std::vector<std::thread> threadsPool(threads);

    if (isColor) {
        for (int i = 0; i < threads - 1; ++i) {
            threadsPool[i] =
                std::thread(&sl::tool::averageTextureRegionColor,
                            std::ref(imgs), std::ref(texture), phaseShiftStep,
                            perRow * i, perRow * (i + 1));
        }
        threadsPool[threads - 1] =
            std::thread(&sl::tool::averageTextureRegionColor, std::ref(imgs),
                        std::ref(texture), phaseShiftStep,
                        perRow * (threads - 1), texture.rows);
    } else {
        for (int i = 0; i < threads - 1; ++i) {
            threadsPool[i] =
                std::thread(&sl::tool::averageTextureRegionGrey, std::ref(imgs),
                            std::ref(texture), phaseShiftStep, perRow * i,
                            perRow * (i + 1));
        }
        threadsPool[threads - 1] =
            std::thread(&sl::tool::averageTextureRegionGrey, std::ref(imgs),
                        std::ref(texture), phaseShiftStep,
                        perRow * (threads - 1), texture.rows);
    }

    for (auto &thread : threadsPool)
        thread.join();
}

void averageTextureRegionGrey(const std::vector<cv::Mat> &imgs,
                              cv::Mat &texture, const int phaseShiftStep,
                              const int rowBegin, const int rowEnd) {
    const int rows = texture.rows;
    const int cols = texture.cols;

    std::vector<const uchar *> ptrImgs(phaseShiftStep);
    for (int i = rowBegin; i < rowEnd; ++i) {
        for (int p = 0; p < phaseShiftStep; ++p)
            ptrImgs[p] = imgs[p].ptr<uchar>(i);
        uchar *ptrTexture = texture.ptr<uchar>(i);
        for (int j = 0; j < cols; ++j) {
            float imgVal = 0.f;
            for (int p = 0; p < phaseShiftStep; ++p)
                imgVal += ptrImgs[p][j];
            imgVal /= phaseShiftStep;
            ptrTexture[j] = static_cast<uchar>(imgVal);
        }
    }
}

void averageTextureRegionColor(const std::vector<cv::Mat> &imgs,
                               cv::Mat &texture, const int phaseShiftStep,
                               const int rowBegin, const int rowEnd) {
    const int rows = texture.rows;
    const int cols = texture.cols;

    std::vector<const cv::Vec3b *> ptrImgs(phaseShiftStep);
    for (int i = rowBegin; i < rowEnd; ++i) {
        for (int p = 0; p < phaseShiftStep; ++p)
            ptrImgs[p] = imgs[p].ptr<cv::Vec3b>(i);
        cv::Vec3b *ptrTexture = texture.ptr<cv::Vec3b>(i);
        for (int j = 0; j < cols; ++j) {
            cv::Vec3f imgVal = 0.f;
            for (int p = 0; p < phaseShiftStep; ++p)
                imgVal += ptrImgs[p][j];
            imgVal /= phaseShiftStep;
            ptrTexture[j] = static_cast<cv::Vec3b>(imgVal);
        }
    }
}

void reverseMappingTexture(const cv::Mat &depth, const cv::Mat &textureIn,
                           const Info &info, cv::Mat &textureAlign) {
    CV_Assert(!depth.empty() && !textureIn.empty() && !info.__M1.empty() &&
              !info.__M3.empty() && !info.__Rlc.empty() && !info.__Tlc.empty());
    CV_Assert(depth.type() == CV_32FC1 || textureIn.type() == CV_8UC3);

    textureAlign =
        cv::Mat(textureIn.size(), textureIn.type(), cv::Scalar(0, 0, 0));

    const int threads = std::thread::hardware_concurrency();
    const int perRow = depth.rows / threads;
    std::vector<std::thread> threadsPool(threads);

    for (int i = 0; i < threads - 1; ++i) {
        threadsPool[i] =
            std::thread(&sl::tool::reverseMappingTextureRegion, std::ref(depth),
                        std::ref(textureIn), std::ref(info),
                        std::ref(textureAlign), perRow * i, perRow * (i + 1));
    }
    threadsPool[threads - 1] =
        std::thread(&sl::tool::reverseMappingTextureRegion, std::ref(depth),
                    std::ref(textureIn), std::ref(info), std::ref(textureAlign),
                    perRow * (threads - 1), depth.rows);

    for (auto &thread : threadsPool)
        thread.join();
}

void reverseMappingTextureRegion(const cv::Mat &depth, const cv::Mat &textureIn,
                                 const Info &info, cv::Mat &textureAlign,
                                 const int rowBegin, const int rowEnd) {
    const int rows = depth.rows;
    const int cols = depth.cols;

    const cv::Mat M1Inv = info.__M1.inv();
    const cv::Mat M3 = info.__M3;
    for (int i = rowBegin; i < rowEnd; ++i) {
        auto ptrDepth = depth.ptr<float>(i);
        auto ptrTextureAlign = textureAlign.ptr<cv::Vec3b>(i);
        for (int j = 0; j < cols; ++j) {
            if (ptrDepth[j] == 0.f)
                continue;

            cv::Mat colorCameraPoint =
                info.__Rlc * M1Inv *
                    (cv::Mat_<double>(3, 1) << j * ptrDepth[j], i * ptrDepth[j],
                     ptrDepth[j]) +
                info.__Tlc;
            cv::Mat imgPoint = M3 * colorCameraPoint;
            const int xMapped =
                imgPoint.at<double>(0, 0) / imgPoint.at<double>(2, 0);
            const int yMapped =
                imgPoint.at<double>(1, 0) / imgPoint.at<double>(2, 0);

            if (xMapped < 0 || xMapped > cols - 1 || yMapped < 0 ||
                yMapped > rows - 1) {
                ptrTextureAlign[j] = cv::Vec3b(0, 0, 0);
                continue;
            }

            ptrTextureAlign[j] = textureIn.ptr<cv::Vec3b>(yMapped)[xMapped];
        }
    }
}

void filterPhase(const cv::Mat &absPhase, cv::Mat &out,
                 const float maxTollerance, const int kernel) {
    CV_Assert(absPhase.type() == CV_32FC1);
    CV_Assert(absPhase.data != out.data);

    out = cv::Mat(absPhase.size(), CV_32FC1, cv::Scalar(0.f));

    const int threads = std::thread::hardware_concurrency();
    const int perRow = absPhase.rows / threads;
    std::vector<std::thread> threadsPool(threads);

    for (int i = 0; i < threads - 1; ++i) {
        threadsPool[i] =
            std::thread(&entryFilterPhase, std::ref(absPhase), std::ref(out),
                        perRow * i, perRow * (i + 1), maxTollerance, kernel);
    }
    threadsPool[threads - 1] = std::thread(
        &entryFilterPhase, std::ref(absPhase), std::ref(out),
        perRow * (threads - 1), absPhase.rows, maxTollerance, kernel);

    for (auto &thread : threadsPool)
        thread.join();
}

void entryFilterPhase(const cv::Mat &absPhase, cv::Mat &out, const int rowBeign,
                      const int rowEnd, const float maxTollerance, int kernel) {
    for (size_t i = rowBeign; i < rowEnd; ++i) {
        if (i < kernel / 2 || i > absPhase.rows - kernel / 2) {
            continue;
        }

        auto pAbsPhase = absPhase.ptr<float>(i);
        auto pOut = out.ptr<float>(i);
        for (size_t j = kernel / 2; j < absPhase.cols - kernel / 2; ++j) {
            int diffCount = 0;
            const float diffTolerrance = maxTollerance;
            for (int d = -kernel / 2; d < kernel / 2; ++d) {
                for (int k = -kernel / 2; k < kernel / 2; ++k) {
                    diffCount = std::abs(absPhase.ptr<float>(i + d)[j + k] -
                                         pAbsPhase[j]) > diffTolerrance
                                    ? diffCount + 1
                                    : diffCount;
                }
            }

            pOut[j] = diffCount > (kernel / 2 + 1) * (kernel / 2 + 1)
                          ? 0.f
                          : pAbsPhase[j];
        }
    }
}

void remapToDepthCamera(const cv::Mat &depthIn, const cv::Mat &textureIn,
                        const cv::Mat &Q, const cv::Mat &M, const cv::Mat &R1,
                        cv::Mat &depthRemaped,
                        pcl::PointCloud<pcl::PointXYZRGB> &cloud) {
    CV_Assert(!depthIn.empty() ||
              textureIn.type() == CV_8UC3 || !textureIn.empty() || !Q.empty() ||
              Q.type() == CV_64FC1 || !M.empty() || M.type() == CV_64FC1 ||
              !R1.empty() || R1.type() == CV_64FC1);
    if (depthIn.data == depthRemaped.data) {
        printf("ERROR:remapToDepthCamera() is can't used when depthIn is same "
               "as depthRemaped! \n");
        return;
    }

    depthRemaped = cv::Mat::zeros(depthIn.rows, depthIn.cols, CV_32FC1);
    cloud.points.clear();

    const float f = Q.at<double>(2, 3);
    const float tx = -1.0 / Q.at<double>(3, 2);
    const float cxlr = Q.at<double>(3, 3) * tx;
    const float cx = -1.0 * Q.at<double>(0, 3);
    const float cy = -1.0 * Q.at<double>(1, 3);

    Eigen::Matrix3d MEigen, R1InvEigen;
    cv::cv2eigen(M, MEigen);
    cv::cv2eigen(R1, R1InvEigen);
    R1InvEigen = R1InvEigen.inverse().eval();

    std::vector<std::thread> threads(std::thread::hardware_concurrency());
    const int rowPerThread = depthIn.rows / threads.size();
    std::mutex mutexMap;
    const int channelStep = depthIn.type() == CV_32FC1 ? 1 : 2;
    for (int i = 0; i < threads.size(); ++i) {
        threads[i] = std::thread([&, i] {
            for (int rowIndex = rowPerThread * i;
                 rowIndex < rowPerThread * (i + 1); ++rowIndex) {
                auto ptrDepthIn = depthIn.ptr<float>(rowIndex);
                for (int colIndex = 0; colIndex < depthIn.cols; ++colIndex) {
                    const int elemetnLoc = colIndex * channelStep;

                    if (!ptrDepthIn[elemetnLoc]) {
                        continue;
                    }

                    Eigen::Vector3d pointBefore;
                    const float xLoc = channelStep == 1 ? colIndex : ptrDepthIn[elemetnLoc + 1];
                    pointBefore << ptrDepthIn[elemetnLoc] * (xLoc - cx) / f,
                        ptrDepthIn[elemetnLoc] * (rowIndex - cy) / f,
                        ptrDepthIn[elemetnLoc];
                    Eigen::Vector3d pointRemaped = R1InvEigen * pointBefore;

                    Eigen::Vector3d pointPixel = MEigen * pointRemaped;

                    const int x_maped =
                        std::round(pointPixel(0, 0) / pointPixel(2, 0));
                    const int y_maped =
                        std::round(pointPixel(1, 0) / pointPixel(2, 0));

                    if ((0 > x_maped) || (y_maped > depthRemaped.rows - 1) ||
                        (0 > y_maped) || (x_maped > depthRemaped.cols - 1))
                        continue;

                    auto color = textureIn.ptr<cv::Vec3b>(y_maped)[x_maped];
                    std::lock_guard<std::mutex> lock(mutexMap);
                    depthRemaped.ptr<float>(y_maped)[x_maped] =
                        pointRemaped(2, 0);
                    cloud.points.emplace_back(pcl::PointXYZRGB(
                        pointRemaped(0, 0), pointRemaped(1, 0),
                        pointRemaped(2, 0), color[0], color[1], color[2]));
                }
            }
        });
    }

    for (auto &thread : threads) {
        thread.join();
    }

    cloud.width = cloud.points.size();
    cloud.height = 1;
    cloud.is_dense = false;
}
} // namespace tool
} // namespace sl