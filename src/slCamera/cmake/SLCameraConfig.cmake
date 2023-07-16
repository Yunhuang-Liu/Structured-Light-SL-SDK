##################################
#   Find SLCamera
##################################
#   This sets the following variables:
# SLCamera_FOUND             -True if SLCamera Was found
# SLCamera_INCLUDE_DIRS      -Directories containing the SLCamera include files
# SLCamera_LIBRARY           -Libraries needed to use SLCamera

find_path(
    SLCamera_INCLUDE_DIRS
    slCameraFactory.h
    ${SLCamera_DIR}/include
)

find_library(
    SLCamera
    libslCamera.so
    ${SLCamera_DIR}/lib
)


find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

set(SLCamera_INCLUDE_DIRS ${SLCamera_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
set(SLCamera_LIBRARIES ${SLCamera} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

find_path(
    withCuda
    cudaTypeDef.cuh
    ${SLCamera_DIR}/include
)

if(${withCuda} STREQUAL ${SLCamera_DIR}/include)
    message("SLCamera with cuda...")
    add_definitions(-D__WITH_CUDA__)
    find_package(CUDA REQUIRED)
    list(APPEND SLCamera_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
    list(APPEND SLCamera_LIBRARIES ${CUDA_LIBRARIES})
endif()

find_path(
    BinoocularCamera_INCLUDE_DIRS
    binoocularCamera.h
    ${SLCamera_DIR}/include
)

if(${BinoocularCamera_INCLUDE_DIRS} STREQUAL ${SLCamera_DIR}/include) 
    message("SLCamera with binocularCamera...")

    find_package(Camera REQUIRED)
    find_package(Projector REQUIRED)

    list(APPEND SLCamera_INCLUDE_DIRS  ${Camera_INCLUDE_DIRS} ${Projector_INCLUDE_DIRS})

    find_library(
        binoocularCamera
        libbinoocularCamera.so
        ${SLCamera_DIR}/lib
    )
    
    list(APPEND SLCamera_LIBRARIES ${binoocularCamera} ${Camera_LIBRARIES} ${Projector_LIBRARIES})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    SLCamera
    DEFAULT_MSG
    SLCamera_INCLUDE_DIRS
    SLCamera_LIBRARIES
)

mark_as_advanced(
    SLCamera_LIBRARIES
    SLCamera_INCLUDE_DIRS
)
