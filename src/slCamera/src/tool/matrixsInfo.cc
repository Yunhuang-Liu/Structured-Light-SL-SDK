#include <matrixsInfo.h>

namespace sl {
namespace tool {
MatrixsInfo::MatrixsInfo(const std::string intrinsicDir,
                         const std::string extrinsicDir) {
    cv::FileStorage readYml(intrinsicDir, cv::FileStorage::READ);
    readYml["M1"] >> __info.__M1;
    readYml["D1"] >> __info.__D1;
    readYml["M2"] >> __info.__M2;
    readYml["D2"] >> __info.__D2;
    readYml["M3"] >> __info.__M3;
    readYml["D3"] >> __info.__D3;
    readYml["M4"] >> __info.__M4;
    readYml["D4"] >> __info.__D4;
    readYml["K1"] >> __info.__K1;
    readYml["K2"] >> __info.__K2;
    readYml.open(extrinsicDir, cv::FileStorage::READ);
    readYml["R1"] >> __info.__R1;
    readYml["P1"] >> __info.__P1;
    readYml["R2"] >> __info.__R2;
    readYml["P2"] >> __info.__P2;
    readYml["Q"] >> __info.__Q;
    readYml["Rlr"] >> __info.__Rlr;
    readYml["Tlr"] >> __info.__Tlr;
    readYml["Rlc"] >> __info.__Rlc;
    readYml["Tlc"] >> __info.__Tlc;
    readYml["Rlp"] >> __info.__Rlp;
    readYml["Tlp"] >> __info.__Tlp;
    readYml["Rrp"] >> __info.__Rrp;
    readYml["Trp"] >> __info.__Trp;
    readYml["S"] >> __info.__S;
    readYml.release();
}

MatrixsInfo::MatrixsInfo(std::string calibrationFileDir) {
    cv::FileStorage readYml(calibrationFileDir, cv::FileStorage::READ);
    readYml["M1"] >> __info.__M1;
    readYml["D1"] >> __info.__D1;
    readYml["M2"] >> __info.__M2;
    readYml["D2"] >> __info.__D2;
    readYml["K1"] >> __info.__K1;
    readYml["Rlr"] >> __info.__Rlr;
    readYml["Tlr"] >> __info.__Tlr;
    readYml["S"] >> __info.__S;
    readYml.release();
}

const Info &MatrixsInfo::getInfo() { return __info; }
} // namespace tool
} // namespace sl