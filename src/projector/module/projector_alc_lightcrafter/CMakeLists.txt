cmake_minimum_required(VERSION 3.0)

project(projectorAlcLightcrafter)

find_package(OpenCV REQUIRED)

set(HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/alcLightCrafter.h)
set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/alcLightCrafter.cc)

include_directories(
    ${CMAKE_SOURCE_DIR}/../common
    ${OpenCV_INCLUDE_DIRS}
)

add_library(
    ${PROJECT_NAME}
    ${HEADERS}
    ${SOURCES}
)

target_link_libraries(
    ${PROJECT_NAME}
    ${OpenCV_LIBRARIES}
)