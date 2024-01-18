execute_process(COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
                OUTPUT_VARIABLE pybind11_PATH
                RESULT_VARIABLE result)

if(result EQUAL 0)
  message(STATUS "pybind11_DIR: '${pybind11_PATH}'")
else()
  message(FATAL_ERROR "Failed to determine pybind11_PATH Error code: ${result}")
endif()

get_filename_component(pybind11_PARENT_DIR ${pybind11_PATH} PATH)
find_package(pybind11 REQUIRED CONFIG PATHS ${pybind11_PARENT_DIR})