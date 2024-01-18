add_library(numpy_cukd SHARED
  numpy_cukd.cu)

target_include_directories(numpy_cukd PUBLIC ${CMAKE_SOURCE_DIR}/include ${pybind11_INCLUDE_DIRS})
target_link_libraries(numpy_cukd PUBLIC
  Python3::Python
  cudart)
target_compile_options(numpy_cukd PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_52,code=sm_52 
-gencode=arch=compute_60,code=compute_60
-gencode=arch=compute_60,code=sm_60
-gencode=arch=compute_61,code=sm_61
-gencode=arch=compute_62,code=sm_62
-gencode=arch=compute_70,code=sm_70
-gencode=arch=compute_75,code=sm_75
-gencode=arch=compute_75,code=compute_75
-gencode=arch=compute_80,code=sm_80
-gencode=arch=compute_86,code=sm_86
-gencode=arch=compute_87,code=sm_87
-gencode=arch=compute_80,code=compute_80
-gencode=arch=compute_86,code=compute_86
-gencode=arch=compute_87,code=compute_87
>)

# set what comes before the library name, here no "lib"
set_target_properties(numpy_cukd PROPERTIES PREFIX "")

file(GLOB module_files "numpy_cukd/*.py")
foreach(file ${module_files})
  configure_file(${file} ${file} COPYONLY)
endforeach()
