/**
 * @file cudaTypeDef.cuh
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief  CUDA类型定义
 * @version 0.1
 * @date 2022-5-9
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __CUDA_CUDATYPEDEFINE_H_
#define __CUDA_CUDATYPEDEFINE_H_

#ifndef CUDACC
#define CUDACC
#endif

#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <Eigen/Eigen>
#include <cuda.h>
#include <cuda/std/cmath>
#include <cuda/std/functional>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudalegacy.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <vector_functions.h>

#endif // __CUDA_CUDATYPEDEFINE_H_
