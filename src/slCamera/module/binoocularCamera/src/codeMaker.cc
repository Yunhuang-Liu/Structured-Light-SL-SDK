#include <codeMaker.h>

#include <fstream>
#include <strstream>

#include <boost/multiprecision/cpp_int.hpp>

namespace elencode {
cv::Mat createBWShiftCode(const int width, const int height, const int blockNum,
                          const int shiftTime) {
    const int blockPixelNum = std::ceil(width / std::pow(2, blockNum));

    cv::Mat img(height, width, CV_8UC1);

    std::vector<uchar> lookTable(blockPixelNum * 2);
    for (size_t i = 0; i < lookTable.size(); ++i) {
        lookTable[i] = (i < lookTable.size() / 2 ? 255 : 0);
    }

    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            //往右移动
            img.ptr<uchar>(i)[j] =
                lookTable[(j + (2 * blockPixelNum - shiftTime)) %
                          (2 * blockPixelNum)];
        }
    }

    return img;
}
} // namespace elencode

GrayEncoder::GrayEncoder(const cv::Size myImgSize)
    : isShiftGrayCode(false), imgSize(myImgSize), isChangeModeY(false),
      isOneHeight(false), isFourFloorGray(false) {}

cv::Mat GrayEncoder::getImage() { return myGrayEncodeImage.clone(); }

cv::Mat GrayEncoder::creatGrayImage(const int numOfImage, const int imgPlainNum,
                                    const int grayBits) {
    if (!isFourFloorGray) {
        if (isChangeModeY) {
            if (!isOneHeight)
                myGrayEncodeImage.create(imgSize.width, imgSize.height, CV_8U);
            else
                myGrayEncodeImage.create(1, imgSize.height, CV_8U);
        } else {
            if (!isOneHeight)
                myGrayEncodeImage.create(imgSize, CV_8U);
            else
                myGrayEncodeImage.create(1, imgSize.width, CV_8U);
        }

        const int mirroTime = imgPlainNum + 1;

        int numOfBlock = (1 << numOfImage) * mirroTime;
        boost::multiprecision::uint1024_t valueOfGrayBits = 0;
        if (numOfImage == 1) {
            valueOfGrayBits = 1;
        } else {
            for (int i = 0; i < (1 << (numOfImage - 2)); i++) {
                boost::multiprecision::uint1024_t six = 6;
                boost::multiprecision::uint1024_t valueAdd = (six << (4 * i));
                valueOfGrayBits = valueOfGrayBits + valueAdd;
            }
        }
        boost::multiprecision::uint1024_t copyPreValue = valueOfGrayBits;
        for (int i = 0; i < imgPlainNum; i++)
            valueOfGrayBits =
                valueOfGrayBits +
                (copyPreValue
                 << (static_cast<int>(std::pow(2, numOfImage)) * (i + 1)));
        boost::multiprecision::uint1024_t valueOfGrayImage = 0;
        for (int i = 0; i < numOfBlock; i++) {
            cv::Mat partImage = myGrayEncodeImage(cv::Rect(
                static_cast<float>(i) / numOfBlock * myGrayEncodeImage.cols, 0,
                myGrayEncodeImage.cols / numOfBlock, myGrayEncodeImage.rows));
            valueOfGrayImage = (valueOfGrayBits >> (numOfBlock - i - 1)) & 1;
            if (valueOfGrayImage == 0) {
                for (int i = 0; i < partImage.rows; i++) {
                    uchar *pimg = partImage.ptr<uchar>(i);
                    for (int j = 0; j < partImage.cols; j++) {
                        pimg[j] = 0;

                        if (1 == numOfImage && isCounterFirstBit) {
                            pimg[j] = 255;
                        }
                    }
                }
            } else {
                for (int i = 0; i < partImage.rows; i++) {
                    uchar *pimg = partImage.ptr<uchar>(i);
                    for (int j = 0; j < partImage.cols; j++) {
                        pimg[j] = 255;

                        if (1 == numOfImage && isCounterFirstBit) {
                            pimg[j] = 0;
                        }
                    }
                }
            }
        }
        if (myGrayEncodeImage.cols % numOfBlock != 0)
            cv::medianBlur(myGrayEncodeImage, myGrayEncodeImage, 3);
        if (isShiftGrayCode) {
            myGrayEncodeImage.copyTo(shiftGrayCodeImage);
            const int pixelShift = static_cast<float>(myGrayEncodeImage.cols) /
                                   (2 * (std::pow(2, grayBits)));
            myGrayEncodeImage(
                cv::Rect(0, 0, pixelShift, myGrayEncodeImage.rows))
                .copyTo(shiftGrayCodeImage(
                    cv::Rect(shiftGrayCodeImage.cols - pixelShift, 0,
                             pixelShift, shiftGrayCodeImage.rows)));
            myGrayEncodeImage(cv::Rect(pixelShift, 0,
                                       myGrayEncodeImage.cols - pixelShift,
                                       myGrayEncodeImage.rows))
                .copyTo(shiftGrayCodeImage(
                    cv::Rect(0, 0, shiftGrayCodeImage.cols - pixelShift,
                             shiftGrayCodeImage.rows)));
            if (isChangeModeY) {
                cv::Mat changeModeYImg(imgSize, CV_8U);
                for (int i = 0; i < shiftGrayCodeImage.rows; i++) {
                    cv::Mat xImg = shiftGrayCodeImage(
                        cv::Rect(0, i, shiftGrayCodeImage.cols, 1));
                    cv::Mat trans = xImg.t();
                    trans.copyTo(changeModeYImg(
                        cv::Rect(i, 0, 1, shiftGrayCodeImage.cols)));
                }
                changeModeYImg.copyTo(shiftGrayCodeImage);
            }
            return shiftGrayCodeImage;
        }
        if (isChangeModeY) {
            if (isOneHeight)
                return myGrayEncodeImage.t();
            cv::Mat changeModeYImg(imgSize, CV_8U);
            for (int i = 0; i < myGrayEncodeImage.rows; i++) {
                cv::Mat xImg = myGrayEncodeImage(
                    cv::Rect(0, i, myGrayEncodeImage.cols, 1));
                cv::Mat trans = xImg.t();
                trans.copyTo(
                    changeModeYImg(cv::Rect(i, 0, 1, myGrayEncodeImage.cols)));
            }
            return changeModeYImg;
        }
        return getImage();
    } else {
        const int blockNum = (numOfImage == 2) ? 16 : 4;
        const cv::Size blockSize =
            cv::Size(imgSize.width / blockNum, imgSize.height);
        cv::Mat color_dark(blockSize, CV_8UC1, cv::Scalar(0));
        cv::Mat color_lightDark(blockSize, CV_8UC1, cv::Scalar(1.f / 3 * 255));
        cv::Mat color_lightWhite(blockSize, CV_8UC1, cv::Scalar(2.f / 3 * 255));
        cv::Mat color_white(blockSize, CV_8UC1, cv::Scalar(1.f * 255));
        std::vector<cv::Mat> floorGray;
        floorGray.push_back(color_dark);
        floorGray.push_back(color_lightDark);
        floorGray.push_back(color_lightWhite);
        floorGray.push_back(color_white);
        myGrayEncodeImage.create(imgSize, CV_8UC1);
        for (int i = 0; i < blockNum; i++) {
            bool isOdd = ((i / 4) % 2 != 0);
            int indexBlock = isOdd ? (3 - i % 4) : i % 4;
            cv::Mat recRegion = myGrayEncodeImage(cv::Rect(
                blockSize.width * i, 0, blockSize.width, blockSize.height));
            floorGray[indexBlock].copyTo(recRegion);
        }
        return getImage();
    }
}

void GrayEncoder::setShiftGrayCodeState(const bool state) {
    isShiftGrayCode = state;
}

void GrayEncoder::setChangeModeY(const bool state) { isChangeModeY = state; }

void GrayEncoder::setIsOneHeightMode(const bool state) { isOneHeight = state; }

void GrayEncoder::setCounterFirstBit(const bool state) {
    isCounterFirstBit = state;
}

void GrayEncoder::setFourFloorMode(const bool state) {
    isFourFloorGray = state;
}

PhaseEncoder::PhaseEncoder(const cv::Size imageSize)
    : isBinaryPhaseCoder(false), isOPWMCoder(false), isCounterMode(false),
      isPhaseErrorExpand(false), isChangeModeY(false), imgSize(imageSize),
      isOneHeight(false), isWhite(false) {}

std::vector<cv::Mat>& PhaseEncoder::getImage() { return myPhaseEncodeImage; }

std::vector<cv::Mat>& PhaseEncoder::creatPhaseImage(const int numOfPhaseShift,
                                      const float patchDistance) {  
    const float pixPerPhaseWeek = patchDistance;
    float x;
    for (int i = 0; i < numOfPhaseShift; i++) {
        cv::Mat phaseImg;
        if (isChangeModeY) {
            if (!isOneHeight)
                phaseImg.create(imgSize.width, imgSize.height,
                                          CV_32FC1);
            else
                phaseImg.create(1, imgSize.height, CV_32FC1);
        } else {
            if (!isOneHeight)
                phaseImg.create(imgSize, CV_32FC1);
            else
                phaseImg.create(1, imgSize.width, CV_32FC1);
        }
        for (int k = 0; k < phaseImg.rows; k++) {
            float *myPtr = phaseImg.ptr<float>(k);
            for (int m = 0; m < phaseImg.cols; m++) {
                x = m * CV_2PI / pixPerPhaseWeek;

                if ((x - std::floor(x / CV_2PI) * CV_2PI) <= CV_PI) {
                    x = x + CV_PI;
                } else {
                    x = x - CV_PI;
                }
                if (isCounterMode) {
                    x = x + CV_PI;
                }
                if (isBinaryPhaseCoder) {
                    float weekValue;
                    if (!isCounterDirection) {
                        weekValue =
                            (x + (i - numOfPhaseShift / 2) * CV_2PI /
                                     numOfPhaseShift) -
                            std::floor((x + (i - numOfPhaseShift / 2) * CV_2PI /
                                                numOfPhaseShift) /
                                       CV_2PI) *
                                CV_2PI;
                    } else {
                        weekValue =
                            (x - (i - numOfPhaseShift / 2) * CV_2PI /
                                     numOfPhaseShift) -
                            std::floor((x - (i - numOfPhaseShift / 2) * CV_2PI /
                                                numOfPhaseShift) /
                                       CV_2PI) *
                                CV_2PI;
                    }
                    if (weekValue < CV_2PI / 4 || weekValue > 3 * CV_2PI / 4) {
                        myPtr[m] = 0;
                    } else {
                        myPtr[m] = 255;
                    }
                } else if (isOPWMCoder) {
                    float weekValue;
                    if (!isCounterDirection) {
                        weekValue =
                            (x + (i - numOfPhaseShift / 2) * CV_2PI /
                                     numOfPhaseShift) -
                            std::floor((x + (i - numOfPhaseShift / 2) * CV_2PI /
                                                numOfPhaseShift) /
                                       CV_2PI) *
                                CV_2PI;
                    } else {
                        weekValue =
                            (x - (i - numOfPhaseShift / 2) * CV_2PI /
                                     numOfPhaseShift) -
                            std::floor((x - (i - numOfPhaseShift / 2) * CV_2PI /
                                                numOfPhaseShift) /
                                       CV_2PI) *
                                CV_2PI;
                    }
                    if ((weekValue > 0.8049 && weekValue < 0.8155) ||
                        (weekValue > 1.0094 && weekValue < 1.1981) ||
                        (weekValue > CV_PI - 0.8155 &&
                         weekValue < CV_PI - 0.8049) ||
                        (weekValue > CV_PI - 1.1981 &&
                         weekValue < CV_PI - 1.0094) ||
                        (weekValue > CV_PI && weekValue < CV_PI + 0.8049) ||
                        (weekValue > CV_PI + 0.8155 &&
                         weekValue < CV_PI + 1.0094) ||
                        (weekValue > CV_PI + 1.1981 &&
                         weekValue < CV_2PI - 1.1981) ||
                        (weekValue > CV_2PI - 1.0094 &&
                         weekValue < CV_2PI - 0.8155) ||
                        (weekValue > CV_2PI - 1.0094 &&
                         weekValue < CV_2PI - 0.8155) ||
                        (weekValue > CV_2PI - 0.8049 && weekValue < CV_2PI)) {
                        myPtr[m] = 0;
                    } else {
                        myPtr[m] = 255;
                    }
                } else if (isWhite) {
                    myPtr[m] = i;
                } else {
                    if (!isCounterDirection) {
                        if (numOfPhaseShift % 2 != 0)
                            myPtr[m] =
                                127.5f +
                                127.5f * cos(x + (i - numOfPhaseShift / 2) *
                                                     CV_2PI / numOfPhaseShift);
                        else
                            myPtr[m] =
                                127.5f +
                                127.5f * cos(x + i * CV_2PI / numOfPhaseShift);
                    } else {
                        if (numOfPhaseShift % 2 != 0)
                            myPtr[m] =
                                127.5f +
                                127.5f * cos(x - (i - numOfPhaseShift / 2) *
                                                     CV_2PI / numOfPhaseShift);
                        else
                            myPtr[m] =
                                127.5f +
                                127.5f * cos(x - i * CV_2PI / numOfPhaseShift);
                    }
                }
            }
        }
        if (isChangeModeY) {
            if (isOneHeight) {
                phaseImg = phaseImg.t();
            } else {
                cv::Mat changeModeYImg(imgSize, CV_32FC1);
                for (int i = 0; i < phaseImg.rows; i++) {
                    cv::Mat xImg = phaseImg(
                        cv::Rect(0, i, phaseImg.cols, 1));
                    cv::Mat trans = xImg.t();
                    trans.copyTo(changeModeYImg(
                        cv::Rect(i, 0, 1, phaseImg.cols)));
                }
                changeModeYImg.copyTo(phaseImg);
            }
        }
        /*
        std::ofstream writeTxtFile("PhaseImage.txt");
        for (int i = 0; i < myPhaseEncodeImage.rows; i++)
        {
                for (int j = 0; j < myPhaseEncodeImage.cols; j++)
                {
                        writeTxtFile << myPhaseEncodeImage.at<float>(i, j)<<" ";
                }
                writeTxtFile << std::endl;
        }
        */
        if (isPhaseErrorExpand) {
            const float threshod = 128;
            for (int k = 0; k < phaseImg.rows; k++) {
                float *ptr_phaseEncodeImage = phaseImg.ptr<float>(k);
                for (int m = 0; m < phaseImg.cols; m++) {
                    const float inputPixel = ptr_phaseEncodeImage[m];
                    ptr_phaseEncodeImage[m] = (inputPixel < threshod ? 0 : 255);
                    const float error = inputPixel - ptr_phaseEncodeImage[m];
                    if (m + 1 < phaseImg.cols) {
                        ptr_phaseEncodeImage[m + 1] += 7.0f / 16.0f * error;
                    }
                    if (k + 1 < phaseImg.rows) {
                        float *ptr_NextRow =
                            phaseImg.ptr<float>(k + 1);
                        ptr_NextRow[m] += 5.0f / 16.0f * error;
                        if (m - 1 >= 0) {
                            ptr_NextRow[m - 1] += 3.0f / 16.0f * error;
                        }
                        if (m + 1 < phaseImg.cols) {
                            ptr_NextRow[m + 1] += 1.0f / 16.0f * error;
                        }
                    }
                }
            }
        }
        phaseImg.convertTo(phaseImg, CV_8U);
        myPhaseEncodeImage.push_back(phaseImg);
    }

    return getImage();
}

void PhaseEncoder::setBinaryPhaseCode(const bool state) {
    isBinaryPhaseCoder = state;
}

void PhaseEncoder::setOPWMCode(const bool state) { isOPWMCoder = state; }

void PhaseEncoder::setCounterMode(const bool state) { isCounterMode = state; }

void PhaseEncoder::setWhiteMode(const bool state) { isWhite = state; }

void PhaseEncoder::setPhaseErrorExpandMode(const bool state) {
    isPhaseErrorExpand = state;
}

void PhaseEncoder::createImgeableImg(const int shiftNum,
                                     const int patchDistance,
                                     std::vector<cv::Mat> &imgs) {
    imgs.resize(shiftNum);
    int pixPerPhaseWeek = patchDistance;
    float x;
    for (int i = 0; i < shiftNum; i++) {
        imgs[i].create(myPhaseEncodeImage[0].size(), CV_32FC1);
        for (int k = 0; k < myPhaseEncodeImage[0].rows; k++) {
            float *myPtr = imgs[i].ptr<float>(k);
            for (int m = 0; m < myPhaseEncodeImage[0].cols; m++) {
                x = m * CV_2PI / pixPerPhaseWeek;
                myPtr[m] = 127.5f +
                           127.5f * cos(x + static_cast<double>(
                                                i - std::floor(shiftNum / 2)) *
                                                CV_2PI / shiftNum);
            }
        }
    }
}

void PhaseEncoder::setChangeModeY(const bool state) { isChangeModeY = state; }

void PhaseEncoder::setIsOneHeightMode(const bool state) { isOneHeight = state; }

void PhaseEncoder::setCounterDirection(const bool state) {
    isCounterDirection = state;
}