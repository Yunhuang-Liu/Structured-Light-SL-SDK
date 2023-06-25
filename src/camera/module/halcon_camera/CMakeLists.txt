cmake_minimum_required(VERSION 3.0)

project(halconCamera)

find_package(OpenCV REQUIRED)

file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
source_group("HEADERS" FILES ${HEADERS})
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)
source_group("SOURCES" FILES ${SOURCES})

include_directories(
    ${CMAKE_SOURCE_DIR}/../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

add_library(
    ${PROJECT_NAME}
    ${HEADERS}
    ${SOURCES}
)

link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

target_link_libraries(
    ${PROJECT_NAME}
    ${OpenCV_LIBRARIES}
)