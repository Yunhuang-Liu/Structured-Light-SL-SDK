# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lyh/桌面/devolope/SL-SDK

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lyh/桌面/devolope/SL-SDK

# Include any dependencies generated for this target.
include src/slCamera/CMakeFiles/slCamera.dir/depend.make

# Include the progress variables for this target.
include src/slCamera/CMakeFiles/slCamera.dir/progress.make

# Include the compile flags for this target's objects.
include src/slCamera/CMakeFiles/slCamera.dir/flags.make

src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_phaseSolver.cu.o: src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_phaseSolver.cu.o.depend
src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_phaseSolver.cu.o: src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_phaseSolver.cu.o.cmake
src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_phaseSolver.cu.o: src/slCamera/src/cuda/phaseSolver.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/lyh/桌面/devolope/SL-SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_phaseSolver.cu.o"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda && /usr/bin/cmake -E make_directory /home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/.
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/./cuda_compile_1_generated_phaseSolver.cu.o -D generated_cubin_file:STRING=/home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/./cuda_compile_1_generated_phaseSolver.cu.o.cubin.txt -P /home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_phaseSolver.cu.o.cmake

src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_rsestructor.cu.o: src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_rsestructor.cu.o.depend
src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_rsestructor.cu.o: src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_rsestructor.cu.o.cmake
src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_rsestructor.cu.o: src/slCamera/src/cuda/rsestructor.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/lyh/桌面/devolope/SL-SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building NVCC (Device) object src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_rsestructor.cu.o"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda && /usr/bin/cmake -E make_directory /home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/.
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/./cuda_compile_1_generated_rsestructor.cu.o -D generated_cubin_file:STRING=/home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/./cuda_compile_1_generated_rsestructor.cu.o.cubin.txt -P /home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_rsestructor.cu.o.cmake

src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_tool.cu.o: src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_tool.cu.o.depend
src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_tool.cu.o: src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_tool.cu.o.cmake
src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_tool.cu.o: src/slCamera/src/cuda/tool.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/lyh/桌面/devolope/SL-SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building NVCC (Device) object src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_tool.cu.o"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda && /usr/bin/cmake -E make_directory /home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/.
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/./cuda_compile_1_generated_tool.cu.o -D generated_cubin_file:STRING=/home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/./cuda_compile_1_generated_tool.cu.o.cubin.txt -P /home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_tool.cu.o.cmake

src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_wrapCreator.cu.o: src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_wrapCreator.cu.o.depend
src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_wrapCreator.cu.o: src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_wrapCreator.cu.o.cmake
src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_wrapCreator.cu.o: src/slCamera/src/cuda/wrapCreator.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/lyh/桌面/devolope/SL-SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building NVCC (Device) object src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_wrapCreator.cu.o"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda && /usr/bin/cmake -E make_directory /home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/.
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/./cuda_compile_1_generated_wrapCreator.cu.o -D generated_cubin_file:STRING=/home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/./cuda_compile_1_generated_wrapCreator.cu.o.cubin.txt -P /home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_wrapCreator.cu.o.cmake

src/slCamera/CMakeFiles/slCamera.dir/common/jsoncpp.cc.o: src/slCamera/CMakeFiles/slCamera.dir/flags.make
src/slCamera/CMakeFiles/slCamera.dir/common/jsoncpp.cc.o: src/slCamera/common/jsoncpp.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lyh/桌面/devolope/SL-SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/slCamera/CMakeFiles/slCamera.dir/common/jsoncpp.cc.o"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/slCamera.dir/common/jsoncpp.cc.o -c /home/lyh/桌面/devolope/SL-SDK/src/slCamera/common/jsoncpp.cc

src/slCamera/CMakeFiles/slCamera.dir/common/jsoncpp.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slCamera.dir/common/jsoncpp.cc.i"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lyh/桌面/devolope/SL-SDK/src/slCamera/common/jsoncpp.cc > CMakeFiles/slCamera.dir/common/jsoncpp.cc.i

src/slCamera/CMakeFiles/slCamera.dir/common/jsoncpp.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slCamera.dir/common/jsoncpp.cc.s"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lyh/桌面/devolope/SL-SDK/src/slCamera/common/jsoncpp.cc -o CMakeFiles/slCamera.dir/common/jsoncpp.cc.s

src/slCamera/CMakeFiles/slCamera.dir/common/slCameraFactory.cc.o: src/slCamera/CMakeFiles/slCamera.dir/flags.make
src/slCamera/CMakeFiles/slCamera.dir/common/slCameraFactory.cc.o: src/slCamera/common/slCameraFactory.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lyh/桌面/devolope/SL-SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/slCamera/CMakeFiles/slCamera.dir/common/slCameraFactory.cc.o"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/slCamera.dir/common/slCameraFactory.cc.o -c /home/lyh/桌面/devolope/SL-SDK/src/slCamera/common/slCameraFactory.cc

src/slCamera/CMakeFiles/slCamera.dir/common/slCameraFactory.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slCamera.dir/common/slCameraFactory.cc.i"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lyh/桌面/devolope/SL-SDK/src/slCamera/common/slCameraFactory.cc > CMakeFiles/slCamera.dir/common/slCameraFactory.cc.i

src/slCamera/CMakeFiles/slCamera.dir/common/slCameraFactory.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slCamera.dir/common/slCameraFactory.cc.s"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lyh/桌面/devolope/SL-SDK/src/slCamera/common/slCameraFactory.cc -o CMakeFiles/slCamera.dir/common/slCameraFactory.cc.s

src/slCamera/CMakeFiles/slCamera.dir/src/phaseSolver/nShiftLineNGrayCodeMasterCpu.cc.o: src/slCamera/CMakeFiles/slCamera.dir/flags.make
src/slCamera/CMakeFiles/slCamera.dir/src/phaseSolver/nShiftLineNGrayCodeMasterCpu.cc.o: src/slCamera/src/phaseSolver/nShiftLineNGrayCodeMasterCpu.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lyh/桌面/devolope/SL-SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object src/slCamera/CMakeFiles/slCamera.dir/src/phaseSolver/nShiftLineNGrayCodeMasterCpu.cc.o"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/slCamera.dir/src/phaseSolver/nShiftLineNGrayCodeMasterCpu.cc.o -c /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/phaseSolver/nShiftLineNGrayCodeMasterCpu.cc

src/slCamera/CMakeFiles/slCamera.dir/src/phaseSolver/nShiftLineNGrayCodeMasterCpu.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slCamera.dir/src/phaseSolver/nShiftLineNGrayCodeMasterCpu.cc.i"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/phaseSolver/nShiftLineNGrayCodeMasterCpu.cc > CMakeFiles/slCamera.dir/src/phaseSolver/nShiftLineNGrayCodeMasterCpu.cc.i

src/slCamera/CMakeFiles/slCamera.dir/src/phaseSolver/nShiftLineNGrayCodeMasterCpu.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slCamera.dir/src/phaseSolver/nShiftLineNGrayCodeMasterCpu.cc.s"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/phaseSolver/nShiftLineNGrayCodeMasterCpu.cc -o CMakeFiles/slCamera.dir/src/phaseSolver/nShiftLineNGrayCodeMasterCpu.cc.s

src/slCamera/CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterCpu.cc.o: src/slCamera/CMakeFiles/slCamera.dir/flags.make
src/slCamera/CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterCpu.cc.o: src/slCamera/src/phaseSolver/nStepNGrayCodeMasterCpu.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lyh/桌面/devolope/SL-SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object src/slCamera/CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterCpu.cc.o"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterCpu.cc.o -c /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/phaseSolver/nStepNGrayCodeMasterCpu.cc

src/slCamera/CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterCpu.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterCpu.cc.i"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/phaseSolver/nStepNGrayCodeMasterCpu.cc > CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterCpu.cc.i

src/slCamera/CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterCpu.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterCpu.cc.s"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/phaseSolver/nStepNGrayCodeMasterCpu.cc -o CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterCpu.cc.s

src/slCamera/CMakeFiles/slCamera.dir/src/rectifier/rectifierCpu.cc.o: src/slCamera/CMakeFiles/slCamera.dir/flags.make
src/slCamera/CMakeFiles/slCamera.dir/src/rectifier/rectifierCpu.cc.o: src/slCamera/src/rectifier/rectifierCpu.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lyh/桌面/devolope/SL-SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object src/slCamera/CMakeFiles/slCamera.dir/src/rectifier/rectifierCpu.cc.o"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/slCamera.dir/src/rectifier/rectifierCpu.cc.o -c /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/rectifier/rectifierCpu.cc

src/slCamera/CMakeFiles/slCamera.dir/src/rectifier/rectifierCpu.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slCamera.dir/src/rectifier/rectifierCpu.cc.i"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/rectifier/rectifierCpu.cc > CMakeFiles/slCamera.dir/src/rectifier/rectifierCpu.cc.i

src/slCamera/CMakeFiles/slCamera.dir/src/rectifier/rectifierCpu.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slCamera.dir/src/rectifier/rectifierCpu.cc.s"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/rectifier/rectifierCpu.cc -o CMakeFiles/slCamera.dir/src/rectifier/rectifierCpu.cc.s

src/slCamera/CMakeFiles/slCamera.dir/src/restructor/restructorCpu.cc.o: src/slCamera/CMakeFiles/slCamera.dir/flags.make
src/slCamera/CMakeFiles/slCamera.dir/src/restructor/restructorCpu.cc.o: src/slCamera/src/restructor/restructorCpu.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lyh/桌面/devolope/SL-SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object src/slCamera/CMakeFiles/slCamera.dir/src/restructor/restructorCpu.cc.o"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/slCamera.dir/src/restructor/restructorCpu.cc.o -c /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/restructor/restructorCpu.cc

src/slCamera/CMakeFiles/slCamera.dir/src/restructor/restructorCpu.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slCamera.dir/src/restructor/restructorCpu.cc.i"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/restructor/restructorCpu.cc > CMakeFiles/slCamera.dir/src/restructor/restructorCpu.cc.i

src/slCamera/CMakeFiles/slCamera.dir/src/restructor/restructorCpu.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slCamera.dir/src/restructor/restructorCpu.cc.s"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/restructor/restructorCpu.cc -o CMakeFiles/slCamera.dir/src/restructor/restructorCpu.cc.s

src/slCamera/CMakeFiles/slCamera.dir/src/restructor/restructorShiftLineCpu.cc.o: src/slCamera/CMakeFiles/slCamera.dir/flags.make
src/slCamera/CMakeFiles/slCamera.dir/src/restructor/restructorShiftLineCpu.cc.o: src/slCamera/src/restructor/restructorShiftLineCpu.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lyh/桌面/devolope/SL-SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object src/slCamera/CMakeFiles/slCamera.dir/src/restructor/restructorShiftLineCpu.cc.o"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/slCamera.dir/src/restructor/restructorShiftLineCpu.cc.o -c /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/restructor/restructorShiftLineCpu.cc

src/slCamera/CMakeFiles/slCamera.dir/src/restructor/restructorShiftLineCpu.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slCamera.dir/src/restructor/restructorShiftLineCpu.cc.i"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/restructor/restructorShiftLineCpu.cc > CMakeFiles/slCamera.dir/src/restructor/restructorShiftLineCpu.cc.i

src/slCamera/CMakeFiles/slCamera.dir/src/restructor/restructorShiftLineCpu.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slCamera.dir/src/restructor/restructorShiftLineCpu.cc.s"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/restructor/restructorShiftLineCpu.cc -o CMakeFiles/slCamera.dir/src/restructor/restructorShiftLineCpu.cc.s

src/slCamera/CMakeFiles/slCamera.dir/src/tool/matrixsInfo.cc.o: src/slCamera/CMakeFiles/slCamera.dir/flags.make
src/slCamera/CMakeFiles/slCamera.dir/src/tool/matrixsInfo.cc.o: src/slCamera/src/tool/matrixsInfo.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lyh/桌面/devolope/SL-SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object src/slCamera/CMakeFiles/slCamera.dir/src/tool/matrixsInfo.cc.o"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/slCamera.dir/src/tool/matrixsInfo.cc.o -c /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/tool/matrixsInfo.cc

src/slCamera/CMakeFiles/slCamera.dir/src/tool/matrixsInfo.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slCamera.dir/src/tool/matrixsInfo.cc.i"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/tool/matrixsInfo.cc > CMakeFiles/slCamera.dir/src/tool/matrixsInfo.cc.i

src/slCamera/CMakeFiles/slCamera.dir/src/tool/matrixsInfo.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slCamera.dir/src/tool/matrixsInfo.cc.s"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/tool/matrixsInfo.cc -o CMakeFiles/slCamera.dir/src/tool/matrixsInfo.cc.s

src/slCamera/CMakeFiles/slCamera.dir/src/tool/tool.cc.o: src/slCamera/CMakeFiles/slCamera.dir/flags.make
src/slCamera/CMakeFiles/slCamera.dir/src/tool/tool.cc.o: src/slCamera/src/tool/tool.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lyh/桌面/devolope/SL-SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object src/slCamera/CMakeFiles/slCamera.dir/src/tool/tool.cc.o"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/slCamera.dir/src/tool/tool.cc.o -c /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/tool/tool.cc

src/slCamera/CMakeFiles/slCamera.dir/src/tool/tool.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slCamera.dir/src/tool/tool.cc.i"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/tool/tool.cc > CMakeFiles/slCamera.dir/src/tool/tool.cc.i

src/slCamera/CMakeFiles/slCamera.dir/src/tool/tool.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slCamera.dir/src/tool/tool.cc.s"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/tool/tool.cc -o CMakeFiles/slCamera.dir/src/tool/tool.cc.s

src/slCamera/CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorCpu.cc.o: src/slCamera/CMakeFiles/slCamera.dir/flags.make
src/slCamera/CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorCpu.cc.o: src/slCamera/src/wrapCreator/wrapCreatorCpu.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lyh/桌面/devolope/SL-SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object src/slCamera/CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorCpu.cc.o"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorCpu.cc.o -c /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/wrapCreator/wrapCreatorCpu.cc

src/slCamera/CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorCpu.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorCpu.cc.i"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/wrapCreator/wrapCreatorCpu.cc > CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorCpu.cc.i

src/slCamera/CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorCpu.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorCpu.cc.s"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/wrapCreator/wrapCreatorCpu.cc -o CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorCpu.cc.s

src/slCamera/CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterGpu.cc.o: src/slCamera/CMakeFiles/slCamera.dir/flags.make
src/slCamera/CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterGpu.cc.o: src/slCamera/src/phaseSolver/nStepNGrayCodeMasterGpu.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lyh/桌面/devolope/SL-SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Building CXX object src/slCamera/CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterGpu.cc.o"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterGpu.cc.o -c /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/phaseSolver/nStepNGrayCodeMasterGpu.cc

src/slCamera/CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterGpu.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterGpu.cc.i"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/phaseSolver/nStepNGrayCodeMasterGpu.cc > CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterGpu.cc.i

src/slCamera/CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterGpu.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterGpu.cc.s"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/phaseSolver/nStepNGrayCodeMasterGpu.cc -o CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterGpu.cc.s

src/slCamera/CMakeFiles/slCamera.dir/src/rectifier/rectifierGpu.cc.o: src/slCamera/CMakeFiles/slCamera.dir/flags.make
src/slCamera/CMakeFiles/slCamera.dir/src/rectifier/rectifierGpu.cc.o: src/slCamera/src/rectifier/rectifierGpu.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lyh/桌面/devolope/SL-SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_16) "Building CXX object src/slCamera/CMakeFiles/slCamera.dir/src/rectifier/rectifierGpu.cc.o"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/slCamera.dir/src/rectifier/rectifierGpu.cc.o -c /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/rectifier/rectifierGpu.cc

src/slCamera/CMakeFiles/slCamera.dir/src/rectifier/rectifierGpu.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slCamera.dir/src/rectifier/rectifierGpu.cc.i"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/rectifier/rectifierGpu.cc > CMakeFiles/slCamera.dir/src/rectifier/rectifierGpu.cc.i

src/slCamera/CMakeFiles/slCamera.dir/src/rectifier/rectifierGpu.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slCamera.dir/src/rectifier/rectifierGpu.cc.s"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/rectifier/rectifierGpu.cc -o CMakeFiles/slCamera.dir/src/rectifier/rectifierGpu.cc.s

src/slCamera/CMakeFiles/slCamera.dir/src/restructor/restructorGpu.cc.o: src/slCamera/CMakeFiles/slCamera.dir/flags.make
src/slCamera/CMakeFiles/slCamera.dir/src/restructor/restructorGpu.cc.o: src/slCamera/src/restructor/restructorGpu.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lyh/桌面/devolope/SL-SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_17) "Building CXX object src/slCamera/CMakeFiles/slCamera.dir/src/restructor/restructorGpu.cc.o"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/slCamera.dir/src/restructor/restructorGpu.cc.o -c /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/restructor/restructorGpu.cc

src/slCamera/CMakeFiles/slCamera.dir/src/restructor/restructorGpu.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slCamera.dir/src/restructor/restructorGpu.cc.i"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/restructor/restructorGpu.cc > CMakeFiles/slCamera.dir/src/restructor/restructorGpu.cc.i

src/slCamera/CMakeFiles/slCamera.dir/src/restructor/restructorGpu.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slCamera.dir/src/restructor/restructorGpu.cc.s"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/restructor/restructorGpu.cc -o CMakeFiles/slCamera.dir/src/restructor/restructorGpu.cc.s

src/slCamera/CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorGpu.cc.o: src/slCamera/CMakeFiles/slCamera.dir/flags.make
src/slCamera/CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorGpu.cc.o: src/slCamera/src/wrapCreator/wrapCreatorGpu.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lyh/桌面/devolope/SL-SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_18) "Building CXX object src/slCamera/CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorGpu.cc.o"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorGpu.cc.o -c /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/wrapCreator/wrapCreatorGpu.cc

src/slCamera/CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorGpu.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorGpu.cc.i"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/wrapCreator/wrapCreatorGpu.cc > CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorGpu.cc.i

src/slCamera/CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorGpu.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorGpu.cc.s"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lyh/桌面/devolope/SL-SDK/src/slCamera/src/wrapCreator/wrapCreatorGpu.cc -o CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorGpu.cc.s

# Object files for target slCamera
slCamera_OBJECTS = \
"CMakeFiles/slCamera.dir/common/jsoncpp.cc.o" \
"CMakeFiles/slCamera.dir/common/slCameraFactory.cc.o" \
"CMakeFiles/slCamera.dir/src/phaseSolver/nShiftLineNGrayCodeMasterCpu.cc.o" \
"CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterCpu.cc.o" \
"CMakeFiles/slCamera.dir/src/rectifier/rectifierCpu.cc.o" \
"CMakeFiles/slCamera.dir/src/restructor/restructorCpu.cc.o" \
"CMakeFiles/slCamera.dir/src/restructor/restructorShiftLineCpu.cc.o" \
"CMakeFiles/slCamera.dir/src/tool/matrixsInfo.cc.o" \
"CMakeFiles/slCamera.dir/src/tool/tool.cc.o" \
"CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorCpu.cc.o" \
"CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterGpu.cc.o" \
"CMakeFiles/slCamera.dir/src/rectifier/rectifierGpu.cc.o" \
"CMakeFiles/slCamera.dir/src/restructor/restructorGpu.cc.o" \
"CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorGpu.cc.o"

# External object files for target slCamera
slCamera_EXTERNAL_OBJECTS = \
"/home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_phaseSolver.cu.o" \
"/home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_rsestructor.cu.o" \
"/home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_tool.cu.o" \
"/home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_wrapCreator.cu.o"

src/slCamera/libslCamera.a: src/slCamera/CMakeFiles/slCamera.dir/common/jsoncpp.cc.o
src/slCamera/libslCamera.a: src/slCamera/CMakeFiles/slCamera.dir/common/slCameraFactory.cc.o
src/slCamera/libslCamera.a: src/slCamera/CMakeFiles/slCamera.dir/src/phaseSolver/nShiftLineNGrayCodeMasterCpu.cc.o
src/slCamera/libslCamera.a: src/slCamera/CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterCpu.cc.o
src/slCamera/libslCamera.a: src/slCamera/CMakeFiles/slCamera.dir/src/rectifier/rectifierCpu.cc.o
src/slCamera/libslCamera.a: src/slCamera/CMakeFiles/slCamera.dir/src/restructor/restructorCpu.cc.o
src/slCamera/libslCamera.a: src/slCamera/CMakeFiles/slCamera.dir/src/restructor/restructorShiftLineCpu.cc.o
src/slCamera/libslCamera.a: src/slCamera/CMakeFiles/slCamera.dir/src/tool/matrixsInfo.cc.o
src/slCamera/libslCamera.a: src/slCamera/CMakeFiles/slCamera.dir/src/tool/tool.cc.o
src/slCamera/libslCamera.a: src/slCamera/CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorCpu.cc.o
src/slCamera/libslCamera.a: src/slCamera/CMakeFiles/slCamera.dir/src/phaseSolver/nStepNGrayCodeMasterGpu.cc.o
src/slCamera/libslCamera.a: src/slCamera/CMakeFiles/slCamera.dir/src/rectifier/rectifierGpu.cc.o
src/slCamera/libslCamera.a: src/slCamera/CMakeFiles/slCamera.dir/src/restructor/restructorGpu.cc.o
src/slCamera/libslCamera.a: src/slCamera/CMakeFiles/slCamera.dir/src/wrapCreator/wrapCreatorGpu.cc.o
src/slCamera/libslCamera.a: src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_phaseSolver.cu.o
src/slCamera/libslCamera.a: src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_rsestructor.cu.o
src/slCamera/libslCamera.a: src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_tool.cu.o
src/slCamera/libslCamera.a: src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_wrapCreator.cu.o
src/slCamera/libslCamera.a: src/slCamera/CMakeFiles/slCamera.dir/build.make
src/slCamera/libslCamera.a: src/slCamera/CMakeFiles/slCamera.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lyh/桌面/devolope/SL-SDK/CMakeFiles --progress-num=$(CMAKE_PROGRESS_19) "Linking CXX static library libslCamera.a"
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && $(CMAKE_COMMAND) -P CMakeFiles/slCamera.dir/cmake_clean_target.cmake
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/slCamera.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/slCamera/CMakeFiles/slCamera.dir/build: src/slCamera/libslCamera.a

.PHONY : src/slCamera/CMakeFiles/slCamera.dir/build

src/slCamera/CMakeFiles/slCamera.dir/clean:
	cd /home/lyh/桌面/devolope/SL-SDK/src/slCamera && $(CMAKE_COMMAND) -P CMakeFiles/slCamera.dir/cmake_clean.cmake
.PHONY : src/slCamera/CMakeFiles/slCamera.dir/clean

src/slCamera/CMakeFiles/slCamera.dir/depend: src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_phaseSolver.cu.o
src/slCamera/CMakeFiles/slCamera.dir/depend: src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_rsestructor.cu.o
src/slCamera/CMakeFiles/slCamera.dir/depend: src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_tool.cu.o
src/slCamera/CMakeFiles/slCamera.dir/depend: src/slCamera/CMakeFiles/cuda_compile_1.dir/src/cuda/cuda_compile_1_generated_wrapCreator.cu.o
	cd /home/lyh/桌面/devolope/SL-SDK && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lyh/桌面/devolope/SL-SDK /home/lyh/桌面/devolope/SL-SDK/src/slCamera /home/lyh/桌面/devolope/SL-SDK /home/lyh/桌面/devolope/SL-SDK/src/slCamera /home/lyh/桌面/devolope/SL-SDK/src/slCamera/CMakeFiles/slCamera.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/slCamera/CMakeFiles/slCamera.dir/depend

