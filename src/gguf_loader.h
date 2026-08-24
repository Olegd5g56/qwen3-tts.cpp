#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <string>
#include <map>
#include <vector>
#include <memory>

namespace qwen3_tts {

// Generic GGUF model loader class
// This is a simplified loader that can be extended for specific model types
class GGUFLoader {
public:
    GGUFLoader();
    ~GGUFLoader();
    
    // Open GGUF file and parse metadata
    bool open(const std::string & path);
    
    // Close file and free resources
    void close();
    
    // Get error message if operation failed
    const std::string & get_error() const { return error_msg_; }
    
    // Get number of tensors in file
    int64_t get_n_tensors() const;
    
    // Get tensor name by index
    const char * get_tensor_name(int64_t idx) const;
    
    // Get tensor type by index
    enum ggml_type get_tensor_type(int64_t idx) const;
    
    // Get tensor offset by index
    size_t get_tensor_offset(int64_t idx) const;
    
    // Get tensor size by index
    size_t get_tensor_size(int64_t idx) const;
    
    // Get metadata value (returns -1 if not found)
    int32_t get_u32(const char * key, int32_t default_val = 0) const;
    float get_f32(const char * key, float default_val = 0.0f) const;
    
    // Get data offset (start of tensor data in file)
    size_t get_data_offset() const;
    
    // Get GGUF context (for advanced usage)
    struct gguf_context * get_ctx() const { return ctx_; }
    
    // Get metadata context
    struct ggml_context * get_meta_ctx() const { return meta_ctx_; }
    
protected:
    struct gguf_context * ctx_ = nullptr;
    struct ggml_context * meta_ctx_ = nullptr;
    std::string error_msg_;
    std::string file_path_;
};

// Helper function to allocate and load tensor data from GGUF file
bool load_tensor_data_from_file(
    const std::string & path,
    struct gguf_context * ctx,
    struct ggml_context * model_ctx,
    const std::map<std::string, struct ggml_tensor *> & tensors,
    ggml_backend_buffer_t & buffer,
    std::string & error_msg,
    enum ggml_backend_dev_type preferred_backend_type = GGML_BACKEND_DEVICE_TYPE_CPU
);

// Helper to initialize backend with GPU preference and CPU fallback.
//
// By default this hands out one process-wide instance, which is what you want
// for stages that run one after another. Pass `exclusive` for a stage that runs
// *concurrently* with another one: a ggml backend instance owns a stream and a
// memory pool, and the CUDA pool is a bump allocator that asserts frees happen
// in reverse order, so two threads sharing one instance will trip it.
ggml_backend_t init_preferred_backend(const char * component_name, std::string * error_msg,
                                      bool exclusive = false);
void release_preferred_backend(ggml_backend_t backend);

// Mark every matmul in `gf` as GGML_PREC_F32.
//
// Both audio models here store their conv weights as F16, and ggml_conv_1d
// lowers to im2col + mul_mat keeping the im2col in F16 too. With an F16 left
// operand the CUDA/HIP backends pick a half-precision cuBLAS compute type, so a
// deep convolution tower accumulates thousands of products in ~3 significant
// digits and the error compounds layer over layer. Ask for an F32 accumulator
// instead; on hardware without tensor cores it is also the faster path.
//
// Call it on a finished graph, before allocating it. Talker graphs deliberately
// do not use this: F16 accumulation is the right default for LLM-shaped matmuls.
void force_f32_matmuls(struct ggml_cgraph * gf);

// Set the thread count on `backend` if and only if it is a CPU backend.
// No-op for GPU/accelerator backends (which ignore thread count). Safe to call
// with a null backend or non-positive n_threads.
void apply_n_threads_to_backend(ggml_backend_t backend, int32_t n_threads);

// Helper function to free model resources
void free_ggml_resources(struct ggml_context * ctx, ggml_backend_buffer_t buffer);

} // namespace qwen3_tts
