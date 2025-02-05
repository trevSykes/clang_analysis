# Usage
# mkdir build
# cd build
# cmake .. -DLLVM_DIR=/opt/homebrew/Cellar/llvm/19.1.7/lib/cmake/llvm -DClang_DIR=/opt/homebrew/Cellar/llvm/19.1.7/lib/cmake/clang
# make

cmake_minimum_required(VERSION 3.4.3)
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.2")
project(TensorCompAnalysis)

#===============================================================================
# 1. LOAD LLVM CONFIGURATION
#===============================================================================
# Set this to a valid LLVM installation dir
set(CT_Clang_INSTALL_DIR "/opt/homebrew/Cellar/llvm/19.1.7" CACHE PATH "LLVM installation directory")

# Add the location of ClangConfig.cmake to CMake search paths (so that
# find_package can locate it)
list(APPEND CMAKE_PREFIX_PATH "${CT_Clang_INSTALL_DIR}/lib/cmake/clang/")

find_package(Clang REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
find_package(jsoncpp CONFIG REQUIRED)


# OperatorAnalysis includes headers from LLVM and Clang - update the include paths
# accordingly
include_directories(SYSTEM "${LLVM_INCLUDE_DIRS};${CLANG_INCLUDE_DIRS};/Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk/usr/include/c++/v1/;/opt/homebrew/Cellar/jsoncpp/1.9.6/include")

#===============================================================================
# 2. LLVM BUILD CONFIGURATION
#===============================================================================
# Use the same C++ standard as LLVM does
set(CMAKE_CXX_STANDARD 20 CACHE STRING "")

# LLVM is normally built without RTTI. Be consistent with that.
if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fno-rtti")
endif()


#===============================================================================
# 3. ADD SOURCE FILES AND TARGETS
#===============================================================================
## Add the hello.cpp file to the project
#add_library(HelloPlugin SHARED hello.cpp)
#
## Link the Clang libraries to the plugin
#target_link_libraries(HelloPlugin PRIVATE clangTooling clangBasic clangASTMatchers)
#
# Add the source file for the plugin

# add_executable(tca-tool
#     main.cpp
#     tensorCompAnalysis.cpp)
add_library(tca-tool
    SHARED
    tensorCompAnalysis.cpp)

target_link_libraries(tca-tool
    PRIVATE
    clangTooling
    clangBasic
    clangAST
    clangFrontend
    jsoncpp_lib)