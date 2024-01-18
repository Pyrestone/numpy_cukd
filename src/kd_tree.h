#pragma once
#include "cukd/builder.h"
#include "cukd/fcp.h"
#include "cukd/knn.h"
#include <math.h>
#include <cuda_runtime.h>

// Copyright Marc Uecker (MIT License)

const int KDTree_MAX_NEIGHBORS = 64;

#define make_noise() \
    { printf("%s called\n", __PRETTY_FUNCTION__); }
#define make_noise() \
    {}

// #define USE_EXPLICIT_DIM 1

#define CUDA_CHECK_ERROR(call)                                                                            \
    {                                                                                                     \
        cudaError_t status = call;                                                                        \
        if (status != cudaSuccess) {                                                                      \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
            exit(1);                                                                                      \
        }                                                                                                 \
    }

struct BasicPoint {
    float3 position;
    uint32_t index;
    operator const float3() const { return this->position; }
#ifdef USE_EXPLICIT_DIM
    uint8_t split_dim;
#endif
};

struct BasicPoint_traits : public cukd::default_data_traits<float3> {
    using point_t = float3;

    static inline __device__ __host__ const point_t& get_point(const BasicPoint& p) { return p.position; }
    static inline __device__ __host__ float get_coord(const BasicPoint& p, int d) { return cukd::get_coord(p.position, d); }
    static inline __device__ __host__ float& get_coord(BasicPoint& p, int d) { return cukd::get_coord(p.position, d); }

#ifdef USE_EXPLICIT_DIM
    enum { has_explicit_dim = true };
    static inline __device__ int get_dim(const BasicPoint& p) { return p.split_dim; }
    static inline __device__ void set_dim(BasicPoint& p, int dim) { p.split_dim = dim; }
#endif
};

int roundUpToNearestPowerOf2(int num) {
    int power = static_cast<int>(std::ceil(std::log2(num)));
    return static_cast<int>(std::pow(2, power));
}

int get_grid_size(int total, int block_size) { return (total + block_size - 1) / block_size; }

__global__ void fill_index_arange(BasicPoint* data, uint32_t size) {
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= size)
        return;
    data[tidx].index = tidx;
}

__global__ void indirect_index(const int* index, const size_t num_indices, const int* values, const size_t num_values, int* result) {
    size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx > num_indices)
        return;
    int idx = index[tidx];
    if (idx < 0 || idx >= num_values) {
        result[tidx] = 0;
    } else {
        result[tidx] = values[idx];
    }
    // printf("result is %d\n",result[tidx]);
}

__global__ void query_fcp(const float3* query_points, BasicPoint* data, const int num_points, const int num_queries, float radius, int* index_of_closest_point) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= num_queries) {
        return;
    }
    float3 query_point = query_points[tidx];

    cukd::FcpSearchParams params{};
    params.cutOffRadius = radius;

    int result_idx = cukd::stackBased::fcp<BasicPoint, BasicPoint_traits>(query_point, data, num_points, params);
    if (result_idx != -1) {
        int return_idx = data[result_idx].index;
        index_of_closest_point[tidx] = return_idx;
    } else {
        index_of_closest_point[tidx] = -1;
    }
}

__global__ void radius_neighbors(const BasicPoint* data, const int num_points, const float radius, int* neighbors) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= num_points) {
        return;
    }
    float3 query_point = data[tidx].position;

    int out_index = data[tidx].index;

    using CandidateList = cukd::HeapCandidateList<KDTree_MAX_NEIGHBORS>;
    CandidateList result(radius);

    cukd::stackBased::knn<CandidateList, BasicPoint, BasicPoint_traits>(result, query_point, data, num_points);

    int num_neighbors = 0;
    for (int i = 0; i < KDTree_MAX_NEIGHBORS; ++i) {
        int r = result.get_pointID(i);
        if (r >= 0) {
            r = data[r].index;
            neighbors[out_index * (KDTree_MAX_NEIGHBORS + 1) + 1 + num_neighbors] = r;
            num_neighbors += 1;
        }
    }
    neighbors[out_index * (KDTree_MAX_NEIGHBORS + 1)] = num_neighbors;
}

__global__ void radius_neighbors_distance_based(const BasicPoint* data, const int num_points, const float radius_multiplier, int* neighbors) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= num_points) {
        return;
    }
    float3 query_point = data[tidx].position;

    int out_index = data[tidx].index;

    float radius = radius_multiplier * (query_point.x * query_point.x + query_point.y * query_point.y + query_point.z * query_point.z);

    using CandidateList = cukd::HeapCandidateList<KDTree_MAX_NEIGHBORS>;
    CandidateList result(radius);

    cukd::stackBased::knn<CandidateList, BasicPoint, BasicPoint_traits>(result, query_point, data, num_points);

    int num_neighbors = 0;
    for (int i = 0; i < KDTree_MAX_NEIGHBORS; ++i) {
        int r = result.get_pointID(i);
        if (r >= 0) {
            r = data[r].index;
        }
        neighbors[out_index * (KDTree_MAX_NEIGHBORS + 1) + 1 + num_neighbors] = r;
        num_neighbors += static_cast<int>(r >= 0);
    }
    neighbors[out_index * (KDTree_MAX_NEIGHBORS + 1)] = num_neighbors;
}

class Immovable {
   public:
    Immovable() = default;
    // Delete copy constructor
    Immovable(const Immovable&) = delete;

    // Delete move constructor
    Immovable(Immovable&&) = delete;

    // Delete copy assignment operator
    Immovable& operator=(const Immovable&) = delete;

    // Delete move assignment operator
    Immovable& operator=(Immovable&&) = delete;
};

template <typename T>
class cudaContainer : public Immovable {
   private:
    T* host_ptr = nullptr;
    T* device_ptr = nullptr;

    size_t num_points = 0;
    size_t num_bytes = 0;

    size_t buffer_size = 0;
    size_t buffer_bytes = 0;

    // we don't own the stream.
    cudaStream_t stream = 0;

   public:
    cudaContainer(size_t num_points, cudaStream_t stream) {
        make_noise();
        // buffer_size should always be >= num_points.
        this->num_points = num_points;
        this->num_bytes = this->num_points * sizeof(T);

        this->buffer_size = roundUpToNearestPowerOf2(num_points);
        this->buffer_bytes = this->buffer_size * sizeof(T);

        assert(this->buffer_bytes >= this->num_bytes);

        this->stream = stream;
    }

    ~cudaContainer() {
        make_noise();
        dealloc_host();
        dealloc_gpu();
    }

    cudaContainer(cudaContainer&& other) {
        make_noise();
        this->host_ptr = other.host_ptr;
        this->device_ptr = other.device_ptr;

        this->num_points = other.num_points;
        this->num_bytes = other.num_bytes;

        this->buffer_size = other.buffer_size;
        this->buffer_bytes = other.buffer_bytes;

        this->stream = other.stream;

        other.device_ptr = nullptr;
        other.host_ptr = nullptr;
        other.num_points = 0;
        other.num_bytes = 0;
        other.buffer_size = 0;
        other.buffer_bytes = 0;
        other.stream = 0;
    }

    cudaContainer& operator=(cudaContainer&& other) {
        make_noise();
        this->dealloc_host();
        this->dealloc_gpu();

        this->host_ptr = other.host_ptr;
        this->device_ptr = other.device_ptr;

        this->num_points = other.num_points;
        this->num_bytes = other.num_bytes;

        this->buffer_size = other.buffer_size;
        this->buffer_bytes = other.buffer_bytes;

        this->stream = other.stream;

        other.device_ptr = nullptr;
        other.host_ptr = nullptr;
        other.num_points = 0;
        other.num_bytes = 0;
        other.buffer_size = 0;
        other.buffer_bytes = 0;
        other.stream = 0;
        return *this;
    }

    size_t size() const { return this->num_points; }

    void alloc_host() {
        make_noise();
        assert(this->host_ptr == nullptr);
        CUDA_CHECK_ERROR(cudaMallocHost(&(this->host_ptr), this->buffer_bytes));
    }

    void dealloc_host() {
        make_noise();
        if (this->host_ptr != nullptr)
            CUDA_CHECK_ERROR(cudaFreeHost(this->host_ptr));
        this->host_ptr = nullptr;
    }

    void alloc_gpu() {
        make_noise();
        assert(this->device_ptr == nullptr);
        CUDA_CHECK_ERROR(cudaMallocAsync(&(this->device_ptr), this->buffer_bytes, this->stream));
    }

    void dealloc_gpu() {
        make_noise();
        if (this->device_ptr != nullptr)
            CUDA_CHECK_ERROR(cudaFreeAsync(this->device_ptr, this->stream));
        this->device_ptr = nullptr;
    }

    T* on_gpu() {
        if (this->device_ptr == nullptr) {
            this->alloc_gpu();
        }
        return this->device_ptr;
    }

    const T* on_gpu() const {
        assert(this->device_ptr != nullptr);
        return this->device_ptr;
    }

    T* on_cpu() {
        if (this->host_ptr == nullptr) {
            this->alloc_host();
        }
        return this->host_ptr;
    }

    const T* on_cpu() const {
        assert(this->host_ptr != nullptr);
        return this->host_ptr;
    }

    T& operator[](const std::ptrdiff_t i) {
        assert(this->host_ptr != nullptr);
        return this->host_ptr[i];
    }

    const T& operator[](const std::ptrdiff_t i) const {
        assert(this->host_ptr != nullptr);
        return this->host_ptr[i];
    }

    void sync() {
        make_noise();
        CUDA_CHECK_ERROR(cudaStreamSynchronize(this->stream));
    }

    cudaStream_t get_stream() { return this->stream; }

    void to_gpu() {
        make_noise();
        assert(this->host_ptr != nullptr);
        assert(this->num_bytes > 0);
        CUDA_CHECK_ERROR(cudaMemcpyAsync(this->on_gpu(), this->on_cpu(), this->num_bytes, cudaMemcpyHostToDevice, stream));
    }

    void to_cpu() {
        make_noise();
        assert(this->device_ptr != nullptr);
        assert(this->num_bytes > 0);
        CUDA_CHECK_ERROR(cudaMemcpyAsync(this->on_cpu(), this->on_gpu(), this->num_bytes, cudaMemcpyDeviceToHost, stream));
    }

    void to_gpu_from(const T* data, size_t size) {
        make_noise();
        assert(size == this->num_points);
        CUDA_CHECK_ERROR(cudaMemcpyAsync(this->on_gpu(), data, this->num_bytes, cudaMemcpyHostToDevice, stream));
    }

    void from_gpu_to(T* data, size_t size) {
        make_noise();
        assert(size == this->num_points);
        CUDA_CHECK_ERROR(cudaMemcpyAsync(data, this->on_cpu(), this->num_bytes, cudaMemcpyDeviceToHost, stream));
    }

    void resize(size_t num_points) {
        make_noise();
        if (this->num_points >= num_points) {
            // resize down, we're fine.
        } else if (this->buffer_size >= num_points) {
            // still fine. we just use more of our buffer.
        } else {
            // num_points is greater than our buffer size.
            // try to re-allocate up to the nearest power of 2 above num_points
            this->dealloc_host();
            this->dealloc_gpu();
            this->buffer_size = roundUpToNearestPowerOf2(num_points);
            this->buffer_bytes = buffer_size * sizeof(T);
        }
        this->num_points = num_points;
        this->num_bytes = num_points * sizeof(T);
        assert(buffer_bytes >= num_bytes);
    }
};

using KDTreeQuery = cudaContainer<float3>;
using KDTreeQueryResult = cudaContainer<int32_t>;

class KDTree3D : public Immovable {
    /*
    This class is more or less a standard container class managing pointers to device and host memory to allow for safe accesses.
    However, since we assume that this will be resized often, we allocate buffers which are rounded up to the nearest power of two.
    After an initial settling period, this should result in very rare re-allocation of the underlying memory.

    // the canonical way of doing this is typically this order:
    1. call constructor
    2. fill in host memory by accessing on_cpu()
    3. call to_gpu()
    4. call build()
    5. get a Query by calling get_query()
    6. fill in the Query's data by accessing it's on_cpu() method.
    7. move the Qu
    resize(), then fill the host data, then call to_gpu, then call build(), then create a QueryHandle and call query() with that handle.
    */
   private:
    BasicPoint* host_ptr = nullptr;
    BasicPoint* device_ptr = nullptr;
    size_t num_points = 0;
    size_t num_bytes = 0;

    // buffer_size should always be >= num_points.
    size_t buffer_size = 0;
    size_t buffer_bytes = 0;
    cudaStream_t stream = 0;
    bool stream_is_owned = true;
    bool built = false;

   public:
    size_t size() { return num_points; }

    KDTree3D(size_t num_points = 0) {
        make_noise();
        this->stream_is_owned = true;
        CUDA_CHECK_ERROR(cudaStreamCreate(&this->stream));
        this->num_points = num_points;
        this->buffer_size = roundUpToNearestPowerOf2(num_points);
        this->num_bytes = this->num_points * sizeof(BasicPoint);
        this->buffer_bytes = this->buffer_size * sizeof(BasicPoint);
        this->built = false;
    };

    KDTree3D(size_t num_points, cudaStream_t stream) {
        make_noise();
        this->stream = stream;
        this->stream_is_owned = false;
        this->num_points = num_points;
        this->buffer_size = roundUpToNearestPowerOf2(num_points);
        this->num_bytes = this->num_points * sizeof(BasicPoint);
        this->buffer_bytes = this->buffer_size * sizeof(BasicPoint);
        this->built = false;
    };

    ~KDTree3D() {
        make_noise();
        this->dealloc_host();
        this->dealloc_gpu();
        if (this->stream_is_owned) {
            CUDA_CHECK_ERROR(cudaStreamDestroy(this->stream));
        }
        this->stream = 0;
    }

    KDTree3D(KDTree3D&& other) {
        make_noise();
        this->host_ptr = other.host_ptr;
        this->device_ptr = other.device_ptr;

        this->num_points = other.num_points;
        this->num_bytes = other.num_bytes;

        this->buffer_size = other.buffer_size;
        this->buffer_bytes = other.buffer_bytes;

        this->stream = other.stream;
        this->built = other.built;
        this->stream_is_owned = other.stream_is_owned;

        other.device_ptr = nullptr;
        other.host_ptr = nullptr;
        other.num_points = 0;
        other.num_bytes = 0;
        other.buffer_size = 0;
        other.buffer_bytes = 0;
        other.stream = 0;
        other.built = false;
        other.stream_is_owned = false;
    }

    KDTree3D& operator=(KDTree3D&& other) {
        make_noise();
        this->dealloc_host();
        this->dealloc_gpu();
        if (this->stream_is_owned) {
            cudaStreamDestroy(this->stream);
        }

        this->host_ptr = other.host_ptr;
        this->device_ptr = other.device_ptr;

        this->num_points = other.num_points;
        this->num_bytes = other.num_bytes;

        this->buffer_size = other.buffer_size;
        this->buffer_bytes = other.buffer_bytes;

        this->stream = other.stream;
        this->built = other.built;
        this->stream_is_owned = other.stream_is_owned;

        other.device_ptr = nullptr;
        other.host_ptr = nullptr;
        other.num_points = 0;
        other.num_bytes = 0;
        other.buffer_size = 0;
        other.buffer_bytes = 0;
        other.stream = 0;
        other.built = false;
        other.stream_is_owned = false;
        return *this;
    }

    void sync() {
        make_noise();
        CUDA_CHECK_ERROR(cudaStreamSynchronize(this->stream));
    }

    void alloc_host() {
        make_noise();
        assert(this->host_ptr == nullptr);
        CUDA_CHECK_ERROR(cudaMallocHost(&(this->host_ptr), this->buffer_bytes));
    }

    void dealloc_host() {
        make_noise();
        if (this->host_ptr != nullptr)
            CUDA_CHECK_ERROR(cudaFreeHost(this->host_ptr));
        this->host_ptr = nullptr;
    }

    void alloc_gpu() {
        make_noise();
        assert(this->device_ptr == nullptr);
        CUDA_CHECK_ERROR(cudaMallocAsync(&(this->device_ptr), this->buffer_bytes, this->stream));
    }

    void dealloc_gpu() {
        make_noise();
        if (this->device_ptr != nullptr)
            CUDA_CHECK_ERROR(cudaFreeAsync(this->device_ptr, this->stream));
        this->device_ptr = nullptr;
    }

    BasicPoint* on_gpu() {
        if (this->device_ptr == nullptr) {
            alloc_gpu();
        }
        return this->device_ptr;
    }

    BasicPoint* on_cpu() {
        if (this->host_ptr == nullptr) {
            alloc_host();
        }
        return this->host_ptr;
    }

    cudaStream_t get_stream() { return this->stream; }

    void to_gpu() {
        make_noise();
        assert(this->host_ptr != nullptr);
        assert(this->num_bytes > 0);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        CUDA_CHECK_ERROR(cudaMemcpyAsync(this->on_gpu(), this->on_cpu(), this->num_bytes, cudaMemcpyHostToDevice, stream));
    }

    void to_cpu() {
        make_noise();
        assert(this->device_ptr != nullptr);
        assert(this->num_bytes > 0);
        CUDA_CHECK_ERROR(cudaMemcpyAsync(this->on_cpu(), this->on_gpu(), this->num_bytes, cudaMemcpyDeviceToHost, stream));
    }

    void resize(size_t num_points) {
        make_noise();
        if (this->num_points >= num_points) {
            // resize down, we're fine.
        } else if (this->buffer_size >= num_points) {
            // still fine. we just use more of our buffer.
        } else {
            // num_points is greater than our buffer size.
            // try to re-allocate up to the nearest power of 2 above num_points
            this->dealloc_host();
            this->dealloc_gpu();
            this->buffer_size = roundUpToNearestPowerOf2(num_points);
            this->buffer_bytes = buffer_size * sizeof(BasicPoint);
        }
        this->num_points = num_points;
        this->num_bytes = num_points * sizeof(BasicPoint);
        // set built to false to indicate that our memory was invalidated.
        this->built = false;
    }

    void build() {
        make_noise();
        int block_size = 1024;
        int grid_size = get_grid_size(this->num_points, block_size);

        // you should not need to call build() more than once on the same tree. if you do, weird things will happen.
        // maybe you forgot to call resize() in-between.
        assert(!this->built);

        // you definitely forgot to move your data to GPU if this assert fails.
        assert(this->device_ptr != nullptr);

        fill_index_arange<<<grid_size, block_size, 0, stream>>>(this->on_gpu(), this->num_points);
        CUKD_CUDA_CHECK(cudaStreamSynchronize(stream));
        cukd::buildTree<BasicPoint, BasicPoint_traits>(this->on_gpu(), this->num_points, nullptr, this->stream, cukd::defaultGpuMemResource());
        CUKD_CUDA_CHECK(cudaStreamSynchronize(stream));
        this->built = true;
    }

    KDTreeQueryResult query(const KDTreeQuery& query, float radius) {
        make_noise();
        KDTreeQueryResult result(query.size(), this->stream);
        int block_size = 1024;
        int grid_size = get_grid_size(query.size(), block_size);
        assert(this->built);
        query_fcp<<<grid_size, block_size, 0, stream>>>(query.on_gpu(), this->on_gpu(), this->num_points, query.size(), radius, result.on_gpu());
        return result;
    }

    void query(const KDTreeQuery& query, float radius, KDTreeQueryResult& result) {
        make_noise();
        // don't cross the streams!
        assert(result.get_stream() == this->stream);

        if (result.size() != query.size()) {
            result.resize(query.size());
        }

        int block_size = 1024;
        int grid_size = get_grid_size(query.size(), block_size);
        assert(this->built);
        query_fcp<<<grid_size, block_size, 0, stream>>>(query.on_gpu(), this->on_gpu(), this->num_points, query.size(), radius, result.on_gpu());
    }

    void query_neighbors(float radius, cudaContainer<int>& neighbors_out) {
        make_noise();
        assert(neighbors_out.get_stream() == this->stream);

        if (neighbors_out.size() != this->num_points * (KDTree_MAX_NEIGHBORS + 1)) {
            neighbors_out.resize(this->num_points * (KDTree_MAX_NEIGHBORS + 1));
        }

        int block_size = 1024;
        int grid_size = get_grid_size(this->num_points, block_size);
        assert(this->built);
        radius_neighbors<<<grid_size, block_size, 0, stream>>>(this->device_ptr, this->num_points, radius, neighbors_out.on_gpu());
    }

    void query_neighbors_db(float radius_multiplier, cudaContainer<int>& neighbors_out) {
        make_noise();
        assert(neighbors_out.get_stream() == this->stream);

        if (neighbors_out.size() != this->num_points * (KDTree_MAX_NEIGHBORS + 1)) {
            neighbors_out.resize(this->num_points * (KDTree_MAX_NEIGHBORS + 1));
        }

        int block_size = 1024;
        int grid_size = get_grid_size(this->num_points, block_size);
        assert(this->built);
        radius_neighbors_distance_based<<<grid_size, block_size, 0, stream>>>(this->device_ptr, this->num_points, radius_multiplier, neighbors_out.on_gpu());
    }
};
