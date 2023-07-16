##################################
#   Find Camera
##################################
#   This sets the following variables:
# Camera_FOUND             -True if Camera Was found
# Camera_INCLUDE_DIRS      -Directories containing the Camera include files
# Camera_LIBRARY           -Libraries needed to use Camera

find_path(
    Camera_INCLUDE_DIRS
    cameraFactory.h
    ${Camera_DIR}/include
)

find_library(
    Camera
    libCamera.so
    ${Camera_DIR}/lib
)

set(Camera_INCLUDE_DIRS ${Camera_INCLUDE_DIR})
set(Camera_LIBRARIES ${Camera})

find_path(
    Huaray_Camera_INCLUDE_DIRS
    huarayCamera.h
    ${Camera_DIR}/include
)

if(${Huaray_Camera_INCLUDE_DIRS} STREQUAL ${Camera_DIR}/include) 
    message("Camera with huaray...")

    find_library(
        MVSDKmd
        libMVSDK.so
        ${Camera_DIR}/lib
    )
    find_library(
        huarayCamera
        libhuarayCamera.so
        ${Camera_DIR}/lib
    )
    
    list(APPEND Camera_LIBRARIES ${MVSDKmd} ${huarayCamera})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Camera
    DEFAULT_MSG
    Camera_INCLUDE_DIRS
    Camera_LIBRARIES
)

mark_as_advanced(
    Camera_LIBRARIES
    Camera_INCLUDE_DIRS
)
