cmake_minimum_required(VERSION 3.0)

project(huarayCamera)

#set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
#set(CMAKE_DEBUG_POSTFIX "d")

find_package(OpenCV REQUIRED)

file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
source_group("HEADERS" FILES ${HEADERS})
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)
source_group("SOURCES" FILES ${SOURCES})

add_library(
    ${PROJECT_NAME}
    SHARED
    ${HEADERS}
    ${SOURCES}
    )

target_link_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
    ${OpenCV_LIBRARIES}
    MVSDK
)

install(
    TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/camera/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/camera/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/camera/lib
)

install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/lib/libMVSDK.so
    DESTINATION ${CMAKE_INSTALL_PREFIX}/camera/lib
)

install(
    FILES ${HEADERS}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/camera/include
)