cmake_minimum_required(VERSION 3.0)

project(binoocularCamera CXX)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(Camera REQUIRED)
find_package(Projector REQUIRED)

file(GLOB BINOOCULAR_CAMERA_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB BINOOCULAR_CAMERA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)

set(DEP_INCLUDE_DIRS     
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/cuda ${CMAKE_CURRENT_SOURCE_DIR}/../../include/phaseSolver
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/rectifier ${CMAKE_CURRENT_SOURCE_DIR}/../../include/restructor
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/tool ${CMAKE_CURRENT_SOURCE_DIR}/../../include/wrapCreator
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${Camera_INCLUDE_DIRS}
    ${Projector_INCLUDE_DIRS})
set(DEP_LIBRARIES ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${Camera_LIBRARIES} ${Projector_LIBRARIES})

if(WITH_CUDA)
    enable_language(CUDA)
    list(APPEND DEP_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
    list(APPEND DEP_LIBRARIES ${CUDA_LIBRARIES})

    include_directories(
        ${PROJECT_NAME}
        ${DEP_INCLUDE_DIRS}
    )

    CUDA_COMPILE(CUDA_OBJS ${HEADERGpu} ${SOURCESGpu})
    list(APPEND SOURCES ${CUDA_OBJS} ${SOURCESGpu})
else()
    include_directories(
        ${PROJECT_NAME}
        ${DEP_INCLUDE_DIRS}
    )
endif()

add_library(
    ${PROJECT_NAME}
    SHARED
    ${SOURCES}
    ${BINOOCULAR_CAMERA_HEADERS}
    ${BINOOCULAR_CAMERA_SOURCES}
)

target_link_libraries(
    ${PROJECT_NAME}
    ${DEP_LIBRARIES}
)

install(
    TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/slCamera/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/slCamera/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/slCamera/lib
)

install(
    FILES ${BINOOCULAR_CAMERA_HEADERS}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/slCamera/include
)