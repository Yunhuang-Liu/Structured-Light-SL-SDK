cmake_minimum_required(VERSION 3.0)

project(GoogleTest)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
message("build google_test...")

find_package(OpenCV REQUIRED)
find_package(Camera REQUIRED)
find_package(Projector REQUIRED)
find_package(SLCamera REQUIRED)

add_executable(
    TestHuarayCamera
    ${CMAKE_CURRENT_SOURCE_DIR}/testHuarayCamera.cc
)

target_include_directories(
    TestHuarayCamera
    PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${Camera_INCLUDE_DIRS}
)

target_link_libraries(
    TestHuarayCamera
    ${OpenCV_LIBRARIES}
    gtest_main
    ${Camera_LIBRARIES}
)

add_executable(
    TestProjectorDlpcApi4710
    ${CMAKE_CURRENT_SOURCE_DIR}/testProjectorDlpcApi4710.cc
)

target_include_directories(
    TestProjectorDlpcApi4710
    PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${Projector_INCLUDE_DIRS}
)

target_link_libraries(
    TestProjectorDlpcApi4710
    ${OpenCV_LIBRARIES}
    gtest_main
    ${Projector_LIBRARIES}
)

add_executable(
    TestProjectorDlpcApi3010
    ${CMAKE_CURRENT_SOURCE_DIR}/testProjectorDlpcApi3010.cc
)

target_include_directories(
    TestProjectorDlpcApi3010
    PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${Projector_INCLUDE_DIRS}
)

target_link_libraries(
    TestProjectorDlpcApi3010
    ${OpenCV_LIBRARIES}
    gtest_main
    ${Projector_LIBRARIES}
)

find_package(SLCamera REQUIRED)

add_executable(
    TestSLCameraCpu
    ${CMAKE_CURRENT_SOURCE_DIR}/testSLCameraCpu.cc
)

target_include_directories(
    TestSLCameraCpu
    PUBLIC
    ${SLCamera_INCLUDE_DIRS}
)

target_link_libraries(
    TestSLCameraCpu
    ${SLCamera_LIBRARIES}
    gtest_main
)

add_executable(
    TestSLCameraGpu
    ${CMAKE_CURRENT_SOURCE_DIR}/testSLCameraGpu.cc
)

target_include_directories(
    TestSLCameraGpu
    PUBLIC
    ${SLCamera_INCLUDE_DIRS}
)

target_link_libraries(
    TestSLCameraGpu
    ${SLCamera_LIBRARIES}
    gtest_main
)

add_executable(
    TestBinacualrCamera
    ${CMAKE_CURRENT_SOURCE_DIR}/testBinocularCamera.cc
)

target_include_directories(
    TestBinacualrCamera
    PUBLIC
    ${SLCamera_INCLUDE_DIRS}
)

target_link_libraries(
    TestBinacualrCamera
    ${SLCamera_LIBRARIES}
    gtest_main
)

add_executable(
    TestShiftLineCodeMaster
    ${CMAKE_CURRENT_SOURCE_DIR}/testShiftLineCodeMaster.cc
)

target_include_directories(
    TestShiftLineCodeMaster
    PUBLIC
    ${SLCamera_INCLUDE_DIRS}
)

target_link_libraries(
    TestShiftLineCodeMaster
    ${SLCamera_LIBRARIES}
    gtest_main
)

enable_testing()
include(GoogleTest)
gtest_discover_tests(TestProjectorDlpcApi4710)
gtest_discover_tests(TestProjectorDlpcApi3010)
gtest_discover_tests(TestSLCameraCpu)
gtest_discover_tests(TestSLCameraGpu)
gtest_discover_tests(TestShiftLineCodeMaster)