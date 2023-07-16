##################################
#   Find Projector
##################################
#   This sets the following variables:
# Projector_FOUND             -True if Projector Was found
# Projector_INCLUDE_DIRS      -Directories containing the Projector include files
# Projector_LIBRARY           -Libraries needed to use Projector

find_path(
    Projector_INCLUDE_DIRS
    projectorFactory.h
    ${Projector_DIR}/include
)

find_library(
    Projector
    libProjector.so
    ${Projector_DIR}/lib
)

find_package(OpenCV REQUIRED)

set(Projector_INCLUDE_DIRS ${Projector_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
set(Projector_LIBRARIES ${Projector} ${OpenCV_LIBRARIES})

find_path(
    Projector_DlpcApi_INCLUDE_DIRS
    projectorDlpc34xx.h
    ${Projector_DIR}/include
)

if(${Projector_DlpcApi_INCLUDE_DIRS} STREQUAL ${Projector_DIR}/include) 
    message("Projector with dlpc34xx...")

    find_library(
        cyusbserial
        libcyusbserial.so
        ${Projector_DIR}/lib
    )
    find_library(
        projectorDlpcApi
        libprojectorDlpcApi.so
        ${Projector_DIR}/lib
    )
    
    list(APPEND Projector_LIBRARIES ${projectorDlpcApi} ${cyusbserial})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Projector
    DEFAULT_MSG
    Projector_INCLUDE_DIRS
    Projector_LIBRARIES
)

mark_as_advanced(
    Projector_LIBRARIES
    Projector_INCLUDE_DIRS
)
