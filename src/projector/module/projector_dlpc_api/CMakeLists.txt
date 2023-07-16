cmake_minimum_required(VERSION 3.0)

project(projectorDlpcApi)

#set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
#set(CMAKE_DEBUG_POSTFIX "d")

find_package(OpenCV REQUIRED)

find_path(CyUsbSerial_DIR CyUSBSerial.h ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cyusbserial)

file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)

add_library(
    ${PROJECT_NAME}
    SHARED
    ${HEADERS}
    ${SOURCES}
)

target_link_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${CyUsbSerial_DIR}
)
        
target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CyUsbSerial_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    ${PROJECT_NAME}
    ${OpenCV_LIBRARIES}
    cyusbserial
)

install(
    TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/projector/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/projector/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/projector/lib
)

file(GLOB CyUsbSerialHeader ${CyUsbSerial_DIR}/*.h)
file(GLOB CyUsbSerialDll ${CyUsbSerial_DIR}/*.so)
file(GLOB CyUsbSerialLib  ${CyUsbSerial_DIR}/*.a)
install(
    FILES ${CyUsbSerialLib}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/projector/lib
)

install(
    FILES ${CyUsbSerialDll}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/projector/lib
)

install(
    FILES ${CyUsbSerialHeader}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/projector/include
)

install(
    FILES ${HEADERS}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/projector/include
)