cmake_minimum_required(VERSION 3.0)

project(binoocularCamera)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_DEBUG_POSTFIX "d")

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(Camera REQUIRED)
find_package(Projector REQUIRED)

file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)

add_library(
    ${PROJECT_NAME}
    ${HEADERS}
    ${SOURCES}
)
        
target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/cuda 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/phaseSolver 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/rectifier 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/restructor
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/tool 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/wrapCreator
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${Camera_INCLUDE_DIRS}
    ${Projector_INCLUDE_DIRS}
)

target_link_libraries(
    ${PROJECT_NAME}
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
    ${Camera_LIBRARIES}
    ${Projector_LIBRARIES}
)

install(
    TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/slCamera/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/slCamera/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/slCamera/lib
)

install(
    FILES ${HEADERS}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/slCamera/include
)