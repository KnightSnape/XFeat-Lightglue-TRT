#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "lightglue.h"
#include <yaml-cpp/yaml.h>

using namespace nvinfer1;

#if NV_TENSORRT_MAJOR >= 10
class PreallocatedOutputAllocator final : public nvinfer1::IOutputAllocator {
public:
    PreallocatedOutputAllocator(void* memory, uint64_t capacity)
        : memory_(memory), capacity_(capacity) {}

    void* reallocateOutputAsync(const char*, void*, uint64_t size,
                                uint64_t, cudaStream_t) noexcept override {
        return size <= capacity_ ? memory_ : nullptr;
    }

    void notifyShape(const char*, const nvinfer1::Dims& dims) noexcept override {
        dims_ = dims;
    }

    const nvinfer1::Dims& dims() const noexcept { return dims_; }

private:
    void* memory_;
    uint64_t capacity_;
    nvinfer1::Dims dims_{};
};
#endif

#define CUDA_CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            std::cerr << "CUDA failure: " << ret << " at line " << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)


using namespace nvinfer1;

void printBindingsInfo(nvinfer1::ICudaEngine* engine) {
#if NV_TENSORRT_MAJOR >= 10
    int nbBindings = engine->getNbIOTensors();
    for (int i = 0; i < nbBindings; ++i) {
        const char* name = engine->getIOTensorName(i);
        nvinfer1::Dims dims = engine->getTensorShape(name);
        bool isInput = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
#else
    int nbBindings = engine->getNbBindings();
    for (int i = 0; i < nbBindings; ++i) {
        const char* name = engine->getBindingName(i);
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        bool isInput = engine->bindingIsInput(i);
#endif

        std::cout << (isInput ? "[Input] " : "[Output] ") << name << ": (";
        for (int j = 0; j < dims.nbDims; ++j) {
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
    }
}

Lightglue::Lightglue(const std::string config_path, const std::string engine_path):dev(torch::kCUDA)
{
    YAML::Node config = YAML::LoadFile(config_path);

    // Lightglue params
    std::string engineFilePath = engine_path;
    maxMatches = config["max_matches"].as<int>();
    threShold = config["match_threshold"].as<double>();
    inputW = config["image_width"].as<int>();
    inputH = config["image_height"].as<int>();
    top_k = config["max_keypoints"].as<int>();

    // Load and initialize the engine
    loadEngine(engineFilePath);
    context = std::unique_ptr<IExecutionContext, GlueDestroyObjects> (engine->createExecutionContext());
    if (!context) {
        throw std::runtime_error("Failed to create execution context");
    }

    // TensorRT 8 uses binding indices; TensorRT 10 uses tensor names.
#if NV_TENSORRT_MAJOR < 10
    image0_size_Index = engine->getBindingIndex("image0_size");
    image1_size_Index = engine->getBindingIndex("image1_size");
    keypoints_0_Index = engine->getBindingIndex("mkpts0");
    keypoints_1_Index = engine->getBindingIndex("mkpts1");
    descriptors_0_Index = engine->getBindingIndex("feats0");
    descriptors_1_Index = engine->getBindingIndex("feats1");
    matches_Index = engine->getBindingIndex("matches");
    scores_Index = engine->getBindingIndex("scores");
#endif


    int kpt_num = top_k;

    // 输入维度在这里设置
    nvinfer1::Dims dim1;
    dim1.nbDims = 1;      // 1个维度
    dim1.d[0] = 2;        // 值为2

    nvinfer1::Dims dim2;
    dim2.nbDims = 1;      // 1个维度
    dim2.d[0] = 2;        // 值为2

#if NV_TENSORRT_MAJOR >= 10
    if (!context->setInputShape("image0_size", dim1) ||
        !context->setInputShape("image1_size", dim2) ||
        !context->setInputShape("mkpts0", nvinfer1::Dims3{1, kpt_num, 2}) ||
        !context->setInputShape("mkpts1", nvinfer1::Dims3{1, kpt_num, 2}) ||
        !context->setInputShape("feats0", nvinfer1::Dims3{1, kpt_num, 64}) ||
        !context->setInputShape("feats1", nvinfer1::Dims3{1, kpt_num, 64})) {
        throw std::runtime_error("Failed to set LightGlue input shapes");
    }
#else
    context->setBindingDimensions(image0_size_Index, dim1);
    context->setBindingDimensions(image1_size_Index, dim2);
    context->setBindingDimensions(keypoints_0_Index, nvinfer1::Dims3{1, kpt_num, 2});
    context->setBindingDimensions(keypoints_1_Index, nvinfer1::Dims3{1, kpt_num, 2});
    context->setBindingDimensions(descriptors_0_Index, nvinfer1::Dims3{1, kpt_num, 64});
    context->setBindingDimensions(descriptors_1_Index, nvinfer1::Dims3{1, kpt_num, 64});
    assert(context->allInputDimensionsSpecified());
#endif

    kpts_bytes_ = static_cast<size_t>(top_k) * 2 * sizeof(float);
    desc_bytes_ = static_cast<size_t>(top_k) * 64 * sizeof(float);
    matches_bytes_ = static_cast<size_t>(maxMatches) * 2 * sizeof(int64_t);
    scores_bytes_ = static_cast<size_t>(maxMatches) * sizeof(float);
    host_matches_.resize(static_cast<size_t>(maxMatches) * 2);
    host_scores_.resize(static_cast<size_t>(maxMatches));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_imgsize0_), 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_imgsize1_), 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_kpts0_), kpts_bytes_));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_kpts1_), kpts_bytes_));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_desc0_), desc_bytes_));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_desc1_), desc_bytes_));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_matches_), matches_bytes_));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scores_), scores_bytes_));

    const std::vector<float> image_size{
        static_cast<float>(inputW), static_cast<float>(inputH)};
    CUDA_CHECK(cudaMemcpyAsync(d_imgsize0_, image_size.data(), 2 * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_imgsize1_, image_size.data(), 2 * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

Lightglue::~Lightglue()
{
    if (stream_) cudaStreamSynchronize(stream_);
    if (d_imgsize0_) cudaFree(d_imgsize0_);
    if (d_imgsize1_) cudaFree(d_imgsize1_);
    if (d_kpts0_) cudaFree(d_kpts0_);
    if (d_kpts1_) cudaFree(d_kpts1_);
    if (d_desc0_) cudaFree(d_desc0_);
    if (d_desc1_) cudaFree(d_desc1_);
    if (d_matches_) cudaFree(d_matches_);
    if (d_scores_) cudaFree(d_scores_);
    if (stream_) cudaStreamDestroy(stream_);
}

void Lightglue::matching(const std::vector<float>& keypoints1,
                         const std::vector<float>& keypoints2,
                         const std::vector<float>& feats1,
                         const std::vector<float>& feats2,
                         std::vector<MatchPoint>& matches)
{
    const size_t expected_kpts = static_cast<size_t>(top_k) * 2;
    const size_t expected_desc = static_cast<size_t>(top_k) * 64;
    if (keypoints1.size() < expected_kpts || keypoints2.size() < expected_kpts ||
        feats1.size() < expected_desc || feats2.size() < expected_desc) {
        throw std::runtime_error("LightGlue received fewer features than its fixed engine shape");
    }
    matches.clear();

    CUDA_CHECK(cudaMemcpyAsync(d_kpts0_, keypoints1.data(), kpts_bytes_,
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_kpts1_, keypoints2.data(), kpts_bytes_,
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_desc0_, feats1.data(), desc_bytes_,
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_desc1_, feats2.data(), desc_bytes_,
                               cudaMemcpyHostToDevice, stream_));
    
#if NV_TENSORRT_MAJOR >= 10
    PreallocatedOutputAllocator matches_allocator(d_matches_, matches_bytes_);
    PreallocatedOutputAllocator scores_allocator(d_scores_, scores_bytes_);
    if (!context->setTensorAddress("image0_size", d_imgsize0_) ||
        !context->setTensorAddress("image1_size", d_imgsize1_) ||
        !context->setTensorAddress("mkpts0", d_kpts0_) ||
        !context->setTensorAddress("mkpts1", d_kpts1_) ||
        !context->setTensorAddress("feats0", d_desc0_) ||
        !context->setTensorAddress("feats1", d_desc1_) ||
        !context->setTensorAddress("matches", d_matches_) ||
        !context->setTensorAddress("scores", d_scores_) ||
        !context->setOutputAllocator("matches", &matches_allocator) ||
        !context->setOutputAllocator("scores", &scores_allocator) ||
        !context->enqueueV3(stream_)) {
        throw std::runtime_error("TensorRT LightGlue inference failed");
    }
    CUDA_CHECK(cudaMemcpyAsync(host_matches_.data(), d_matches_, matches_bytes_,
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(host_scores_.data(), d_scores_, scores_bytes_,
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
#else
    void* bindings[8];
    bindings[image0_size_Index] = d_imgsize0_;
    bindings[image1_size_Index] = d_imgsize1_;
    bindings[keypoints_0_Index] = d_kpts0_;
    bindings[keypoints_1_Index] = d_kpts1_;
    bindings[descriptors_0_Index] = d_desc0_;
    bindings[descriptors_1_Index] = d_desc1_;
    bindings[matches_Index] = d_matches_;
    bindings[scores_Index] = d_scores_;

    // auto start = std::chrono::high_resolution_clock::now();

    // Run inference on TensorRT engine
    context->enqueueV2(bindings, stream_, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
#endif

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> duration = end - start;
    // std::cout<<"LightGlue Inference benchmark done "<< duration.count() << " ms"<< std::endl;

    // Dims match_dims = context->getBindingDimensions(matches_Index);
#if NV_TENSORRT_MAJOR >= 10
    nvinfer1::Dims match_dims = matches_allocator.dims();
#else
    nvinfer1::Dims match_dims = context->getTensorShape("matches");
#endif
    int num_matches = match_dims.d[0];
    if (match_dims.nbDims != 2 || num_matches < 0 ||
        num_matches > maxMatches) {
        throw std::runtime_error(
            "Invalid LightGlue output shape: matches=" +
            std::to_string(num_matches));
    }

#if NV_TENSORRT_MAJOR < 10
    CUDA_CHECK(cudaMemcpy(host_matches_.data(), d_matches_,
                          num_matches * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_scores_.data(), d_scores_,
                          num_matches * sizeof(float), cudaMemcpyDeviceToHost));
#endif

    // 过滤低分匹配对，并保存
    for (int i = 0; i < num_matches; ++i) {
        if (host_scores_[i] >= threShold){
            matches.push_back({static_cast<int>(host_matches_[i * 2]),
                               static_cast<int>(host_matches_[i * 2 + 1]),
                               host_scores_[i]});
        }
    }
}


std::vector<char> Lightglue::readEngineFile(const std::string& engineFilePath)
{
    std::ifstream file(engineFilePath, std::ios::binary | std::ios::ate);
    if(!file.is_open()){
        throw std::runtime_error("Unable to open engine file: " + engineFilePath);
    }
    std::streamsize size = file.tellg();
    file.seekg(0,std::ios::beg);

    std::vector<char> buffer(size);
    if(!file.read(buffer.data(),size)){
        throw std::runtime_error("Unable to read engine file: " + engineFilePath);
    }
    return buffer;
}

void Lightglue::loadEngine(const std::string& engineFilePath)
{
    std::vector<char> engineData = readEngineFile(engineFilePath);

    runtime = std::unique_ptr<IRuntime,GlueDestroyObjects>(createInferRuntime(gLogger));

    if(!runtime){
        throw std::runtime_error("Unable to create TensorRT runtime");
    }

    bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
    ICudaEngine* rawEngine = runtime->deserializeCudaEngine(engineData.data(),engineData.size());

    // printBindingsInfo(rawEngine);

    if(!rawEngine)
    {
        throw std::runtime_error("Unable to deserialize TensorRT engine");
    }
    engine = std::unique_ptr<ICudaEngine, GlueDestroyObjects>(rawEngine);
}
