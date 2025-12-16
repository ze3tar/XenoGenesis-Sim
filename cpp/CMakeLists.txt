add_library(xenogenesis_ca_stepper OBJECT
  ca/ca_stepper.cpp
  common/rng.cpp
  common/metrics.cpp
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(xenogenesis_ca_stepper PUBLIC OpenMP::OpenMP_CXX)
endif()

if(FFTW3F_FOUND)
  target_compile_definitions(xenogenesis_ca_stepper PUBLIC XG_USE_FFTW)
  target_link_libraries(xenogenesis_ca_stepper PUBLIC FFTW3F::FFTW3F)
endif()

target_include_directories(xenogenesis_ca_stepper
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(xenogenesis_pybind MODULE bindings/pybind_module.cpp $<TARGET_OBJECTS:xenogenesis_ca_stepper>)
set_target_properties(xenogenesis_pybind PROPERTIES PREFIX "" OUTPUT_NAME "xenogenesis_native")

target_include_directories(xenogenesis_pybind PRIVATE ${Python_INCLUDE_DIRS})
if(pybind11_FOUND)
  target_link_libraries(xenogenesis_pybind PRIVATE pybind11::module)
endif()

install(TARGETS xenogenesis_pybind LIBRARY DESTINATION ${Python_SITEARCH})
