cmake_minimum_required(VERSION 3.0)

project(SLSDK)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

option(BUILD_PROJECTOR "build the projector controller lib" OFF)
set(BUILD_Projector_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/projector)
option(BUILD_CAMERA "build the camera controller lib" OFF)
set(BUILD_Camera_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/camera)
option(BUILD_SLSDK "build the structed light camera lib" ON)
set(BUILD_SLSDK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/slCamera)
option(BUILD_GOOGLE_TEST "build the google_test targets" ON)
set(BUILD_GOOGLE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test)

if(BUILD_PROJECTOR)
    add_definitions(-DWITH_PROJECTOR)
    add_subdirectory(${BUILD_Projector_DIR})
    list(APPEND ${DependencyLIBS} Projector) 
    message("build projector...")
endif()

if(BUILD_CAMERA)
    add_subdirectory(${BUILD_Camera_DIR})
    list(APPEND ${DependencyLIBS} Camera)
    message("build camera...")
endif()

if(BUILD_SLSDK)
    add_definitions(-DWITH_SLSDK)
    add_subdirectory(${BUILD_SLSDK_DIR})
    list(APPEND ${DependencyLIBS} SLSDK)
    message("build structed light sdk...")
endif()

if(BUILD_GOOGLE_TEST)
    add_subdirectory(${BUILD_GOOGLE_DIR})
    message("build google_test...")
endif()