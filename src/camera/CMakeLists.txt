cmake_minimum_required(VERSION 3.0)

project(Camera)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_DEBUG_POSTFIX "d")

set(CAMERA_ROOT_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/common/camera.h
                       ${CMAKE_CURRENT_SOURCE_DIR}/common/cameraFactory.h
                       ${CMAKE_CURRENT_SOURCE_DIR}/common/typeDef.h)
set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/common/cameraFactory.cc)

set(INCLUDE_DIRS "")
set(CAMERA_DEP_LIBS "")

find_package(OpenCV REQUIRED)

if(DEFINED BUILD_HALCON_CAMERA)
unset(BUILD_HALCON_CAMERA CACHE)
endif()

set(HALCON_CAMERA_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/module/halcon_camera)
option(BUILD_HALCON_CAMERA "build halcon camera" OFF)
if(BUILD_HALCON_CAMERA)
list(APPEND CAMERA_DEP_LIBS halconCamera)
list(APPEND INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/module/halcon_camera/include)
add_subdirectory(${HALCON_CAMERA_DIRS})
message("build halcon camera...")
endif()

if(DEFINED HUARAY_CAMERA_DIRS)
    unset(HUARAY_CAMERA_DIRS CACHE)
endif()

set(HUARAY_CAMERA_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/module/huaray_camera)
option(BUILD_HUARAY_CAMERA "build huaray camera" ON)
if(BUILD_HUARAY_CAMERA)
    list(APPEND CAMERA_DEP_LIBS huarayCamera)
    list(APPEND INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/module/huaray_camera/include)
    add_subdirectory(${HUARAY_CAMERA_DIRS})
    message("build huaray camera...")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${INCLUDE_DIRS}
)

add_library(
    ${PROJECT_NAME}
    ${CAMERA_ROOT_HEADER}
    ${SOURCES}
)

target_link_libraries(
    ${PROJECT_NAME}
    ${CAMERA_DEP_LIBS}
    ${OpenCV_LIBRARIES}
)

install(
    FILES 
    ${CAMERA_ROOT_HEADER}
    DESTINATION
    ${CMAKE_INSTALL_PREFIX}/camera/include
)

install(
    TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/camera/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/camera/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/camera/lib
)

install(
    FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CameraConfig.cmake
    DESTINATION
    ${CMAKE_INSTALL_PREFIX}/camera
)