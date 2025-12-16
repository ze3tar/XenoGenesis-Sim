cmake_minimum_required(VERSION 3.18)
project(xenogenesis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  message(STATUS "pybind11 not found via config; fetching from PyPI")
  execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --includes OUTPUT_VARIABLE PYBIND11_INCLUDES OUTPUT_STRIP_TRAILING_WHITESPACE)
  separate_arguments(PYBIND11_INCLUDES)
endif()

find_package(FFTW3f QUIET)
find_package(OpenMP QUIET)

add_subdirectory(cpp)
