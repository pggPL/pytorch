#ifndef STABLE_TORCH_SHIM
#define STABLE_TORCH_SHIM

#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <torch/csrc/stable/version.h>

// This header defines stable C API extensions for backward/forward
// compatibility when calling ATen operations through the dispatcher.
//
// This is separate from the main AOTI shim to provide versioning capabilities
// for schema changes in native ATen functions.

#ifdef __cplusplus
extern "C" {
#endif

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0

// Has the same semantic as aoti_torch_call_dispatcher, but takes an
// additional argument for the extension build version. This is
// needed for backward compatibility when calling native functions via
// the dispatcher. The caller should pass in the libtorch version the
// extension is building with (NOT target version).
AOTI_TORCH_EXPORT AOTITorchError torch_call_dispatcher(
    const char* opName,
    const char* overloadName,
    StableIValue* stack,
    uint64_t extension_build_version);

// Version-aware variant of aoti_torch_library_impl that takes an
// extension_build_version parameter for backward compatibility
AOTI_TORCH_EXPORT AOTITorchError torch_library_impl(
    TorchLibraryHandle self,
    const char* name,
    void (*fn)(StableIValue*, uint64_t, uint64_t),
    uint64_t extension_build_version);

struct StableListOpaque;
using StableListHandle = StableListOpaque*;

// returns an owning reference of a StableList. callee is responsible for
// freeing memory.
AOTI_TORCH_EXPORT AOTITorchError
torch_new_list_reserve_size(size_t size, StableListHandle* ret);

AOTI_TORCH_EXPORT AOTITorchError
torch_list_size(StableListHandle list_handle, size_t* size);

AOTI_TORCH_EXPORT AOTITorchError torch_list_get_item(
    StableListHandle list_handle,
    size_t index,
    StableIValue* element);

AOTI_TORCH_EXPORT AOTITorchError torch_list_set_item(
    StableListHandle list_handle,
    size_t index,
    StableIValue element);

AOTI_TORCH_EXPORT AOTITorchError
torch_list_push_back(StableListHandle list_handle, StableIValue element);

// deletes the underlying list referenced by list_handle
AOTI_TORCH_EXPORT AOTITorchError
torch_delete_list(StableListHandle list_handle);

// Helper function to parse device string using c10::Device
// Returns device type and index via output parameters
AOTI_TORCH_EXPORT AOTITorchError torch_parse_device_string(
    const char* device_string,
    uint32_t* out_device_type,
    int32_t* out_device_index);

// Parallel utility APIs for stable ABI
// Function pointer type for parallel_for callback
// The callback receives begin and end indices for a range to process
typedef void (*ParallelFunc)(int64_t begin, int64_t end, void* ctx);

AOTI_TORCH_EXPORT AOTITorchError torch_parallel_for(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    ParallelFunc func,
    void* ctx);

// Get the current thread index in a parallel region
// Returns 0 if not in a parallel region
AOTI_TORCH_EXPORT AOTITorchError torch_get_thread_idx(uint32_t* out_thread_idx);

// Get the number of threads for the parallel backend
AOTI_TORCH_EXPORT AOTITorchError
torch_get_num_threads(uint32_t* out_num_threads);

// Get a pointer to the underlying storage data
AOTI_TORCH_EXPORT AOTITorchError torch_get_mutable_data_ptr(
    AtenTensorHandle tensor,
    void** ret_data_ptr // returns borrowed reference
);

AOTI_TORCH_EXPORT AOTITorchError torch_get_const_data_ptr(
    AtenTensorHandle tensor,
    const void** ret_data_ptr // returns borrowed reference
);

struct StringOpaque;
using StringHandle = StringOpaque*;

AOTI_TORCH_EXPORT AOTITorchError
torch_new_string_handle(const char* data, size_t length, StringHandle* handle);

AOTI_TORCH_EXPORT AOTITorchError torch_delete_string(StringHandle handle);

AOTI_TORCH_EXPORT AOTITorchError
torch_string_length(StringHandle handle, size_t* length);

AOTI_TORCH_EXPORT AOTITorchError
torch_string_c_str(StringHandle handle, const char** data);

#ifdef USE_CUDA

AOTI_TORCH_EXPORT AOTITorchError
torch_get_current_cuda_blas_handle(void** ret_handle);

AOTI_TORCH_EXPORT AOTITorchError
torch_set_current_cuda_stream(void* stream, int32_t device_index);

AOTI_TORCH_EXPORT AOTITorchError torch_get_cuda_stream_from_pool(
    bool isHighPriority,
    int32_t device_index,
    void** ret_stream);

AOTI_TORCH_EXPORT AOTITorchError
torch_cuda_stream_synchronize(void* stream, int32_t device_index);

// Wrapper around c10_cuda_check_implementation that captures the error message
// without propagating the exception. The caller must free error_msg using
// torch_c10_cuda_free_error_msg if it is non-null.
AOTI_TORCH_EXPORT AOTITorchError torch_c10_cuda_check_msg(
    int32_t err,
    const char* filename,
    const char* function_name,
    uint32_t line_number,
    bool include_device_assertions,
    char** error_msg);

// Free error message allocated by torch_c10_cuda_check_msg
AOTI_TORCH_EXPORT void torch_c10_cuda_free_error_msg(char* error_msg);

#endif // USE_CUDA

// Set requires_grad on a tensor
AOTI_TORCH_EXPORT AOTITorchError
torch_set_requires_grad(AtenTensorHandle tensor, bool requires_grad);

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0

/**
 * The beginning of all shims added in 2.11.0 onwards.
 */
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_11_0

// Shims for a few dtypes not already in
// torch/csrc/inductor/aoti_torch/c/shim.h
AOTI_TORCH_EXPORT int32_t torch_dtype_float8_e8m0fnu();
AOTI_TORCH_EXPORT int32_t torch_dtype_float4_e2m1fn_x2();

// Creates a tensor from an existing data blob with an optional deleter.
// The deleter receives both the data pointer and a caller-supplied context
// pointer, which allows passing capturing lambdas across the C ABI boundary
// by heap-allocating the callable and passing it as deleter_ctx.
AOTI_TORCH_EXPORT AOTITorchError torch_from_blob(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret, // returns new reference
    int32_t layout,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size,
    void (*deleter)(void* data, void* ctx),
    void* deleter_ctx);

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_11_0

/**
 * The beginning of all shims added in 2.12.0 onwards.
 */
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_12_0

// Tag getter functions for ABI-stable tag passing.  By hiding these behind
// functions, the precise enum ordinal is NOT part of the ABI contract.
AOTI_TORCH_EXPORT int32_t torch_tag_core();
AOTI_TORCH_EXPORT int32_t torch_tag_cudagraph_unsafe();
AOTI_TORCH_EXPORT int32_t torch_tag_data_dependent_output();
AOTI_TORCH_EXPORT int32_t torch_tag_dynamic_output_shape();
AOTI_TORCH_EXPORT int32_t torch_tag_flexible_layout();
AOTI_TORCH_EXPORT int32_t torch_tag_generated();
AOTI_TORCH_EXPORT int32_t torch_tag_inplace_view();
AOTI_TORCH_EXPORT int32_t torch_tag_maybe_aliasing_or_mutating();
AOTI_TORCH_EXPORT int32_t torch_tag_needs_contiguous_strides();
AOTI_TORCH_EXPORT int32_t torch_tag_needs_exact_strides();
AOTI_TORCH_EXPORT int32_t torch_tag_needs_fixed_stride_order();
AOTI_TORCH_EXPORT int32_t torch_tag_nondeterministic_bitwise();
AOTI_TORCH_EXPORT int32_t torch_tag_nondeterministic_seeded();
AOTI_TORCH_EXPORT int32_t torch_tag_out_variant();
AOTI_TORCH_EXPORT int32_t torch_tag_pointwise();
AOTI_TORCH_EXPORT int32_t torch_tag_pt2_compliant_tag();
AOTI_TORCH_EXPORT int32_t torch_tag_reduction();
AOTI_TORCH_EXPORT int32_t torch_tag_view_copy();

// Stable corollary to torch::Library method m.def() with tags.
// Tags are passed as int32_t values obtained from torch_tag_*() getters,
// not raw enum ordinals, so the ABI is stable across versions.
AOTI_TORCH_EXPORT AOTITorchError torch_library_def_with_tags(
    TorchLibraryHandle self,
    const char* schema,
    const int32_t* tags,
    int32_t num_tags);
#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_12_0

/**
 * The beginning of all shims added in 2.13.0 onwards.
 */
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0

// Stable corollary to torch::Library method m.set_python_module(...).
AOTI_TORCH_EXPORT AOTITorchError torch_library_set_python_module(
    TorchLibraryHandle self,
    const char* pymodule,
    const char* context);

/// Retrieve a pointer to the string that holds the most recent exception's
/// message and backtrace that occurred in the calling thread. This pointer is a
/// borrowed pointer and is invalidated when the next exception occurs or the
/// calling thread is shutdown. This may be the same as the less detailed
/// torch_exception_get_what_without_backtrace() in case more information is not
/// available.
AOTI_TORCH_EXPORT const char* torch_exception_get_what();

/// Retrieve a pointer to the string that holds the most recent exception's
/// message that occurred in the calling thread. This pointer is a borrowed
/// pointer and is invalidated when the next exception occurs or the calling
/// thread is shutdown.
AOTI_TORCH_EXPORT const char* torch_exception_get_what_without_backtrace();

// Allocates an StableIValue on the heap, returns an owning pointer.
// This allocation must be deleted with torch_delete_stable_ivalue or
// by passing it to a dispatch call which frees it internally.
AOTI_TORCH_EXPORT AOTITorchError
torch_new_stable_ivalue(StableIValue** ret_value);

// Frees an StableIValue that was created by torch_new_stable_ivalue.
// Deleting a nullptr is invalid and returns failure, allocations must
// only be deleted once.
AOTI_TORCH_EXPORT AOTITorchError
torch_delete_stable_ivalue(StableIValue* value);

/// Retrieves the underlying Stream's backend-specific non-owning stream handle
/// (e.g. `cudaStream_t` for CUDA). Returns a void* that can be `static_cast`ed
/// accordingly.
AOTI_TORCH_EXPORT AOTITorchError
torch_stream_native_handle(StreamHandle stream, void** ret_native_handle);

// Returns a new owning AtenGeneratorHandle that shares the underlying RNG state
// with `self` (the copy bumps the GeneratorImpl refcount). The callee owns the
// result and must free it with torch_delete_generator.
AOTI_TORCH_EXPORT AOTITorchError torch_new_generator_handle(
    AtenGeneratorHandle self,
    AtenGeneratorHandle* ret_new_generator);

// Frees an owning AtenGeneratorHandle previously returned by
// torch_new_generator_handle (or otherwise handed off with ownership).
AOTI_TORCH_EXPORT AOTITorchError
torch_delete_generator(AtenGeneratorHandle generator);

// Returns the generator's device as (device_type, device_index). device_type
// uses the same encoding as the aoti_torch_device_type_*() getters.
AOTI_TORCH_EXPORT AOTITorchError torch_generator_get_device(
    AtenGeneratorHandle generator,
    int32_t* ret_device_type,
    int32_t* ret_device_index);

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0

/**
 * The beginning of all shims added in 2.14.0 onwards.
 */
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_14_0

// Opaque handle to a heap-allocated at::Scalar. A Scalar is a tagged variant
// (bool / int / double / complex<double>) and therefore cannot use the trivial
// 64-bit StableIValue reinterpret convention that int/double/bool use; it is
// passed by handle instead, mirroring StringHandle. The handle owns the
// underlying Scalar and must be freed with torch_delete_scalar (or handed off
// to a dispatch call, which frees it).
struct ScalarOpaque;
using ScalarHandle = ScalarOpaque*;

// Constructors: return a new owning ScalarHandle. The callee is responsible for
// freeing it with torch_delete_scalar.
AOTI_TORCH_EXPORT AOTITorchError
torch_new_scalar_int(int64_t value, ScalarHandle* ret);
AOTI_TORCH_EXPORT AOTITorchError
torch_new_scalar_double(double value, ScalarHandle* ret);
AOTI_TORCH_EXPORT AOTITorchError
torch_new_scalar_bool(bool value, ScalarHandle* ret);
AOTI_TORCH_EXPORT AOTITorchError
torch_new_scalar_complex_double(double real, double imag, ScalarHandle* ret);

// Discriminators mirroring c10::Scalar's tag. Exactly one of is_bool / is_int /
// is_double / is_complex_double is true for any valid Scalar.
AOTI_TORCH_EXPORT AOTITorchError
torch_scalar_is_int(ScalarHandle handle, bool* ret);
AOTI_TORCH_EXPORT AOTITorchError
torch_scalar_is_double(ScalarHandle handle, bool* ret);
AOTI_TORCH_EXPORT AOTITorchError
torch_scalar_is_bool(ScalarHandle handle, bool* ret);
AOTI_TORCH_EXPORT AOTITorchError
torch_scalar_is_complex_double(ScalarHandle handle, bool* ret);

// Typed getters. The caller is responsible for querying the discriminator
// first; these follow c10::Scalar's own (lossy) cross-type conversion
// semantics.
AOTI_TORCH_EXPORT AOTITorchError
torch_scalar_to_int(ScalarHandle handle, int64_t* ret);
AOTI_TORCH_EXPORT AOTITorchError
torch_scalar_to_double(ScalarHandle handle, double* ret);
AOTI_TORCH_EXPORT AOTITorchError
torch_scalar_to_bool(ScalarHandle handle, bool* ret);
AOTI_TORCH_EXPORT AOTITorchError torch_scalar_to_complex_double(
    ScalarHandle handle,
    double* ret_real,
    double* ret_imag);

// Frees an owning ScalarHandle previously returned by torch_new_scalar_* (or
// otherwise handed off with ownership).
AOTI_TORCH_EXPORT AOTITorchError torch_delete_scalar(ScalarHandle handle);

// Opaque handle to a c10d::ProcessGroup (held by an internal
// c10::intrusive_ptr, so the handle keeps the group alive). Obtained from a
// Python torch.distributed.ProcessGroup via torch_processgroup_from_pyobject
// (see torch/csrc/stable/python/c/shim.h) and freed with
// torch_delete_processgroup. All functions below require a libtorch built with
// USE_DISTRIBUTED; otherwise they return an error.
struct ProcessGroupOpaque;
using ProcessGroupHandle = ProcessGroupOpaque*;

// Opaque handle to a c10d::Work (held by an internal c10::intrusive_ptr). The
// collectives below return a new owning WorkHandle representing the async
// operation. The caller must eventually free it with torch_delete_work; call
// torch_work_wait first to block until the collective completes.
struct WorkOpaque;
using WorkHandle = WorkOpaque*;

// ReduceOp encoding for the collectives below. These values match
// c10d::ReduceOp::RedOpType, so the ordinal is part of the ABI contract:
//   SUM=0, AVG=1, PRODUCT=2, MIN=3, MAX=4, BAND=5, BOR=6, BXOR=7.
// PREMUL_SUM is intentionally unsupported (it needs a scaling factor).

// Frees an owning ProcessGroupHandle. Does not affect the underlying process
// group beyond dropping this reference.
AOTI_TORCH_EXPORT AOTITorchError
torch_delete_processgroup(ProcessGroupHandle pg);

AOTI_TORCH_EXPORT AOTITorchError
torch_processgroup_rank(ProcessGroupHandle pg, int32_t* ret_rank);

AOTI_TORCH_EXPORT AOTITorchError
torch_processgroup_size(ProcessGroupHandle pg, int32_t* ret_size);

// Sets *ret to true iff the group's backend type is NCCL.
AOTI_TORCH_EXPORT AOTITorchError
torch_processgroup_backend_is_nccl(ProcessGroupHandle pg, bool* ret);

// Collectives. Each takes borrowed tensor handles, launches the async op, and
// returns a new owning WorkHandle in *ret_work. reduce_op uses the encoding
// documented above.
AOTI_TORCH_EXPORT AOTITorchError torch_processgroup_allreduce(
    ProcessGroupHandle pg,
    AtenTensorHandle* tensors,
    size_t num_tensors,
    int32_t reduce_op,
    WorkHandle* ret_work);

AOTI_TORCH_EXPORT AOTITorchError torch_processgroup_allreduce_coalesced(
    ProcessGroupHandle pg,
    AtenTensorHandle* tensors,
    size_t num_tensors,
    int32_t reduce_op,
    WorkHandle* ret_work);

AOTI_TORCH_EXPORT AOTITorchError torch_processgroup_broadcast(
    ProcessGroupHandle pg,
    AtenTensorHandle* tensors,
    size_t num_tensors,
    int64_t root_rank,
    WorkHandle* ret_work);

// allgather restricted to the single-input / single-output-group form: gathers
// the one `input` tensor from every rank into `output` (num_output tensors,
// one per rank).
AOTI_TORCH_EXPORT AOTITorchError torch_processgroup_allgather(
    ProcessGroupHandle pg,
    AtenTensorHandle* output,
    size_t num_output,
    AtenTensorHandle input,
    WorkHandle* ret_work);

AOTI_TORCH_EXPORT AOTITorchError
torch_processgroup_barrier(ProcessGroupHandle pg, WorkHandle* ret_work);

// Blocks until the work completes. ret_completed receives the bool returned by
// c10d::Work::wait (whether the work completed; it may also throw on error,
// which is converted to an error code).
AOTI_TORCH_EXPORT AOTITorchError
torch_work_wait(WorkHandle work, bool* ret_completed);

// Frees an owning WorkHandle previously returned by a collective.
AOTI_TORCH_EXPORT AOTITorchError torch_delete_work(WorkHandle work);

#ifdef USE_CUDA

// Returns the number of streaming multiprocessors (SMs) on the CUDA device with
// the given index. Reads cudaDeviceProp::multiProcessorCount for that device.
AOTI_TORCH_EXPORT AOTITorchError torch_cuda_get_device_multiprocessor_count(
    int32_t device_index,
    int32_t* ret_sm_count);

// Returns the CUDA compute capability of the device with the given index as
// separate major and minor components (cudaDeviceProp::major / ::minor). The
// "sm_arch" used by kernel launch config is major * 10 + minor.
AOTI_TORCH_EXPORT AOTITorchError torch_cuda_get_device_compute_capability(
    int32_t device_index,
    int32_t* ret_major,
    int32_t* ret_minor);

// Wraps an externally-owned cudaStream_t in a new owning StreamHandle bound to
// the given device. The returned handle must be freed with
// aoti_torch_delete_stream; doing so frees only the wrapper, not the underlying
// cudaStream_t (its lifetime remains the caller's responsibility). ext_stream
// is a cudaStream_t passed as void*.
AOTI_TORCH_EXPORT AOTITorchError torch_get_cuda_stream_from_external(
    void* ext_stream,
    int32_t device_index,
    StreamHandle* ret_stream);

#endif // USE_CUDA

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_14_0

#ifdef __cplusplus
} // extern "C"
#endif

#endif // STABLE_TORCH_SHIM
