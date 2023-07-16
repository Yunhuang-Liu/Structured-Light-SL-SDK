cmake_minimum_required(VERSION 3.0)

project(Projector)

#set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
#set(CMAKE_DEBUG_POSTFIX "d")

message("build projector...")

set(PROJECTOR_ROOT_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/common/projector.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/common/projectorFactory.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/common/typeDef.h)

set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/common/projectorFactory.cc)
set(PROJECTOR_DEP_LIBS ${OpenCV_LIBRARIES})

find_package(OpenCV REQUIRED)
set(INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/common)

set(PROJECTOR_WITH_DLPCAPI_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/module/projector_dlpc_api)
option(BUILD_PROJECTOR_WITH_DLPCAPI "build projector with dlpc_api" ON)
if(BUILD_PROJECTOR_WITH_DLPCAPI)
    list(APPEND PROJECTOR_DEP_LIBS projectorDlpcApi)
    list(APPEND INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/module/projector_dlpc_api/include)
    add_subdirectory(${PROJECTOR_WITH_DLPCAPI_DIRS})
    message("build projector with dlpc_api...")
endif()

set(PROJECTOR_WITH_ALC_LIGHTCRAFTER_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/module/projector_alc_lightcrafter)
option(BUILD_PROJECTOR_WITH_ALC_LIGHTCRAFTER "build projector with alc lightcrafter" OFF)
if(BUILD_PROJECTOR_WITH_ALC_LIGHTCRAFTER)
    list(APPEND PROJECTOR_DEP_LIBS projectorAlcLightcrafter)
    list(APPEND INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/module/projector_alc_lightcrafter/include)
    add_subdirectory(${PROJECTOR_WITH_ALC_LIGHTCRAFTER_DIRS})
endif()

add_library(
    ${PROJECT_NAME}
    SHARED
    ${PROJECTOR_ROOT_HEADER}
    ${SOURCES}
)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${INCLUDE_DIRS}
)

target_link_libraries(
    ${PROJECT_NAME}
    ${PROJECTOR_DEP_LIBS}
)

install(
    FILES 
    ${PROJECTOR_ROOT_HEADER}
    DESTINATION
    ${CMAKE_INSTALL_PREFIX}/projector/include
)

install(
    TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/projector/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/projector/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/projector/lib
)

install(
    FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ProjectorConfig.cmake
    DESTINATION
    ${CMAKE_INSTALL_PREFIX}/projector
)