cmake_minimum_required(VERSION 3.0)

project(slCamera)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g")
#set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
#set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
#set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")

message("build slCamera...")

file(GLOB_RECURSE SLCamera_ROOT_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/common/*.h
${CMAKE_CURRENT_SOURCE_DIR}/include/matrixsInfo.h ${CMAKE_CURRENT_SOURCE_DIR}/include/phaseSolver.h
${CMAKE_CURRENT_SOURCE_DIR}/include/rectifier.h ${CMAKE_CURRENT_SOURCE_DIR}/include/restructor.h
${CMAKE_CURRENT_SOURCE_DIR}/include/tool.h ${CMAKE_CURRENT_SOURCE_DIR}/include/typeDef.h 
${CMAKE_CURRENT_SOURCE_DIR}/include/wrapCreator.h
${CMAKE_CURRENT_SOURCE_DIR}/include/*Cpu.h)

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/common/jsoncpp.cc
${CMAKE_CURRENT_SOURCE_DIR}/src/tool.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/matrixsInfo.cc
${CMAKE_CURRENT_SOURCE_DIR}/src/*Cpu.cc)

file(GLOB_RECURSE HEADERGpu ${CMAKE_CURRENT_SOURCE_DIR}/include/*Gpu.h ${CMAKE_CURRENT_SOURCE_DIR}/include/*.cuh)

file(GLOB_RECURSE SOURCESGpu ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/src/*Gpu.cc)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

set(SLCamera_DEP_LIBS ${OpenCV_LIBRARIES} ${PCL_LIBRARIES}) 
set(INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/common ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda ${CMAKE_CURRENT_SOURCE_DIR}/include/phaseSolver ${CMAKE_CURRENT_SOURCE_DIR}/include/rectifier ${CMAKE_CURRENT_SOURCE_DIR}/include/restructor
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tool ${CMAKE_CURRENT_SOURCE_DIR}/include/wrapCreator)

include_directories(
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS} 
    ${INCLUDE_DIRS}
)

option(WITH_CUDA "Build the slCamera with cuda" ON)
if(WITH_CUDA) 
    find_package(CUDA REQUIRED)
    if(CUDA_FOUND)
        message("build the slCamera with cuda...")
        add_definitions(-D__WITH_CUDA__)
        list(APPEND INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
        include_directories(
            ${EIGEN3_INCLUDE_DIRS}
            ${OpenCV_INCLUDE_DIRS}
            ${PCL_INCLUDE_DIRS} 
            ${INCLUDE_DIRS}
        )
        list(APPEND SLCamera_DEP_LIBS ${CUDA_LIBRARIES})
        list(APPEND SLCamera_ROOT_HEADER ${HEADERGpu})
        list(APPEND SOURCES ${SOURCESGpu})
    endif()
endif()

set(BinoocularCamera_DIR ${CMAKE_CURRENT_SOURCE_DIR}/module/binoocularCamera)
option(BUILD_BinoocularCamera "build binoocularCamera" ON)
if(BUILD_BinoocularCamera)
    list(APPEND SLCamera_DEP_LIBS binoocularCamera)
    list(APPEND INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/module/binoocularCamera/include)
    add_subdirectory(${BinoocularCamera_DIR})
    message("build binoocularCamera...")
endif()

set(MonocularCamera_DIR ${CMAKE_CURRENT_SOURCE_DIR}/module/monocularCamera)
option(BUILD_MonocularCamera "build monocularCamera" OFF)
if(BUILD_MonocularCamera)
    list(APPEND SLCamera_DEP_LIBS monocularCamera)
    list(APPEND INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/module/monocularCamera/include)
    add_subdirectory(${MonocularCamera_DIR})
endif()

add_library(
    ${PROJECT_NAME}
    SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/common/slCameraFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/common/slCameraFactory.cc
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
    ${SLCamera_DEP_LIBS}
)

install(
    FILES 
    ${SLCamera_ROOT_HEADER}
    DESTINATION
    ${CMAKE_INSTALL_PREFIX}/slCamera/include
)

install(
    TARGETS 
    ${PROJECT_NAME} 
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/slCamera/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/slCamera/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/slCamera/lib
)

install(
    FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SLCameraConfig.cmake
    DESTINATION
    ${CMAKE_INSTALL_PREFIX}/slCamera
)