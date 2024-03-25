#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "kd_tree.h"

// Copyright Marc Uecker (MIT License)

namespace py = pybind11;

__global__ void kernel(){
    printf("Block %d Thread %d reporting\n",blockIdx.x,threadIdx.x);
}

void call_kernel(){
    printf("calling kernel...\n");
    kernel<<<1,32>>>();
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
    printf("done.\n");
}

void test_numpy(pybind11::array_t<float> arr){

    std::cout<<"shape: (";
    for (int dim=0; dim<arr.ndim(); ++dim){
        if(dim>0){
            std::cout<<", ";
        }
        std::cout<<arr.shape(dim);
    }
    std::cout<<")"<<std::endl;
    
    std::cout<<"dtype:"<<arr.dtype()<<std::endl;
    const float* array_data=arr.data();
    std::cout<<"data: [";
    for (int i=0; i<arr.size(); ++i){
        if(i>0){
            std::cout<<", ";
        }
        std::cout<<array_data[i];
    }
    std::cout<<"]"<<std::endl;
    call_kernel();
}

py::capsule create_kdtree(const size_t size=0){
    KDTree3D* tree = new KDTree3D(size);
    return py::capsule(tree,[](void* tree){
        delete ((KDTree3D*)tree);
    });
}

void build_tree(py::capsule tree, const py::array_t<float, py::array::c_style | py::array::forcecast> points){
    KDTree3D* t = tree.get_pointer<KDTree3D>();
    if(points.ndim()!=2){
        throw std::runtime_error("points.ndim must be 2!");
    }
    if(points.shape(1)!=3){
        throw std::runtime_error("points.shape[1] must be 3!");
    }
    size_t n=points.shape(0);
    t->resize(n);
    BasicPoint* data_ptr=t->on_cpu();
    auto points_access=points.unchecked<2>();
    for (int i=0; i<n; ++i){
        data_ptr[i].position.x=points_access(i,0);
        data_ptr[i].position.y=points_access(i,1);
        data_ptr[i].position.z=points_access(i,2);
    }
    t->to_gpu();
    t->build();
    t->sync();
}

py::capsule make_tree(const py::array_t<float, py::array::c_style | py::array::forcecast> points){
    if(points.ndim()!=2){
        throw std::runtime_error("points.ndim must be 2!");
    }
    if(points.shape(1)!=3){
        throw std::runtime_error("points.shape[1] must be 3!");
    }
    size_t n=points.shape(0);
    py::capsule tree=create_kdtree(n);
    build_tree(tree, points);
    return tree;
}


py::capsule create_query(const py::capsule tree, const size_t size=0){
    KDTree3D* t = tree.get_pointer<KDTree3D>();
    KDTreeQuery* query = new KDTreeQuery(size,t->get_stream());
    return py::capsule(query,[](void* query){
        delete ((KDTreeQuery*)query);
    });
}

py::capsule create_result(const py::capsule tree, const size_t size=0){
    KDTree3D* t = tree.get_pointer<KDTree3D>();
    KDTreeQueryResult* result = new KDTreeQueryResult(size,t->get_stream());
    return py::capsule(result,[](void* result){
        delete ((KDTreeQueryResult*)result);
    });
}


py::array_t<int32_t> query_tree(py::capsule tree, const py::array_t<float, py::array::c_style | py::array::forcecast> points, float radius, py::handle query = py::none(), py::handle result = py::none()){
    if(points.ndim()!=2){
        throw std::runtime_error("points.ndim must be 2!");
    }
    if(points.shape(1)!=3){
        throw std::runtime_error("points.shape[1] must be 3!");
    }
    KDTree3D* tree_ = tree.get_pointer<KDTree3D>();
    bool created_query = false;
    bool created_result = false;
    KDTreeQuery* query_=nullptr;
    KDTreeQueryResult* result_=nullptr;

    const size_t n_query=points.shape(0);

    if(query.is_none()){
        query_=new KDTreeQuery(0,tree_->get_stream());
        created_query=true;
    } else {
        if(py::isinstance<py::capsule>(query)){
            py::capsule caps= query.cast<py::capsule>();
            query_ = caps.get_pointer<KDTreeQuery>();
        }
    }

    if(result.is_none()){
        result_=new KDTreeQueryResult(0,tree_->get_stream());
        created_query=true;
    } else {
        if(py::isinstance<py::capsule>(result)){
            py::capsule caps= result.cast<py::capsule>();
            result_ = caps.get_pointer<KDTreeQueryResult>();
        }
    }

    constexpr size_t elsize=sizeof(int32_t);
    const size_t shape[1]{n_query};
    const size_t strides[1]{elsize};

    py::array_t<int32_t> out_array =py::array_t<int32_t, py::array::c_style | py::array::forcecast>(shape,strides);
    


    query_->resize(n_query);
    result_->resize(n_query);
    float3* query_data = query_->on_cpu();

    const float* points_data = points.data();

    for(int i=0; i<n_query; ++i){
        query_data[i].x=points_data[i*3+0];
        query_data[i].y=points_data[i*3+1];
        query_data[i].z=points_data[i*3+2];
    }
    query_->to_gpu();
    query_->sync();
    tree_->query(*query_,radius,*result_);
    result_->to_cpu();
    tree_->sync();
    /* std::cout<<"C++ query result: [";
    for(int i=0; i<n_query; ++i){
        if(i>0) std::cout<<", ";
        std::cout<<result_->on_cpu()[i];
    }
    std::cout<<"]"<<std::endl; */

    const int32_t* result_data=result_->on_cpu();
    int32_t* out_ptr=out_array.mutable_data();
    for(int i=0; i<n_query; ++i){
        out_ptr[i]=result_data[i];
    }

    if (created_query){
        delete query_;
    }
    if (created_result){
        delete result_;
    }

    return out_array;
}


PYBIND11_MODULE(numpy_cukd, m)
{
  m.def("test_numpy", test_numpy);
  m.def("create_kdtree",create_kdtree);
  m.def("make_tree",make_tree);
  m.def("build_tree",build_tree);
  m.def("create_query",create_query);
  m.def("create_result",create_result);
  m.def("query_tree",query_tree);
}