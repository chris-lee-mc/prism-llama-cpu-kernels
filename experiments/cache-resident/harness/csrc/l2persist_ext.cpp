// Persisting-L2 control for torch load(). Mirrors harness/l2persist.py; pointers and handles are int64.
#include <torch/extension.h>
#include <cuda_runtime.h>

#include <cstring>
#include <tuple>
#include <vector>

#define CR_CHECK(call)                                                          \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        TORCH_CHECK(err_ == cudaSuccess, #call " failed: ", cudaGetErrorName(err_), " (", \
                    cudaGetErrorString(err_), ")");                             \
    } while (0)

static int dev_attr(cudaDeviceAttr a, int dev) {
    int v = 0;
    CR_CHECK(cudaDeviceGetAttribute(&v, a, dev));
    return v;
}

static int64_t max_persisting_bytes(int64_t dev) { return dev_attr(cudaDevAttrMaxPersistingL2CacheSize, (int) dev); }
static int64_t max_window_bytes(int64_t dev)     { return dev_attr(cudaDevAttrMaxAccessPolicyWindowSize, (int) dev); }
static int64_t l2_bytes(int64_t dev)             { return dev_attr(cudaDevAttrL2CacheSize, (int) dev); }

static int64_t get_limit() {
    size_t v = 0;
    CR_CHECK(cudaDeviceGetLimit(&v, cudaLimitPersistingL2CacheSize));
    return (int64_t) v;
}

static void set_limit(int64_t nbytes) {
    CR_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, (size_t) nbytes));
}

static void reset() { CR_CHECK(cudaCtxResetPersistingL2Cache()); }

static cudaAccessProperty prop_of(int64_t p) {
    TORCH_CHECK(p >= 0 && p <= 2, "bad access property ", p);
    return (cudaAccessProperty) p;
}

static cudaAccessPolicyWindow make_window(int64_t ptr, int64_t nbytes, double hit_ratio, int64_t hit_prop, int64_t miss_prop) {
    cudaAccessPolicyWindow w;
    memset(&w, 0, sizeof(w));
    w.base_ptr  = (void *) (intptr_t) ptr;
    w.num_bytes = (size_t) nbytes;
    w.hitRatio  = (float) hit_ratio;
    w.hitProp   = prop_of(hit_prop);
    w.missProp  = prop_of(miss_prop);
    return w;
}

static void set_stream_window(int64_t stream, int64_t ptr, int64_t nbytes, double hit_ratio, int64_t hit_prop, int64_t miss_prop) {
    cudaStreamAttrValue v;
    memset(&v, 0, sizeof(v));
    v.accessPolicyWindow = make_window(ptr, nbytes, hit_ratio, hit_prop, miss_prop);
    CR_CHECK(cudaStreamSetAttribute((cudaStream_t) (intptr_t) stream, cudaStreamAttributeAccessPolicyWindow, &v));
}

// returns [base_ptr, num_bytes, hit_prop, miss_prop], hit ratio as second return
static std::tuple<std::vector<int64_t>, double> get_stream_window(int64_t stream) {
    cudaStreamAttrValue v;
    memset(&v, 0, sizeof(v));
    CR_CHECK(cudaStreamGetAttribute((cudaStream_t) (intptr_t) stream, cudaStreamAttributeAccessPolicyWindow, &v));
    const cudaAccessPolicyWindow & w = v.accessPolicyWindow;
    std::vector<int64_t> out = {(int64_t) (intptr_t) w.base_ptr, (int64_t) w.num_bytes, (int64_t) w.hitProp, (int64_t) w.missProp};
    return std::make_tuple(out, (double) w.hitRatio);
}

static std::vector<cudaGraphNode_t> kernel_nodes(int64_t graph) {
    cudaGraph_t g = (cudaGraph_t) (intptr_t) graph;
    size_t n = 0;
    CR_CHECK(cudaGraphGetNodes(g, nullptr, &n));
    std::vector<cudaGraphNode_t> nodes(n);
    if (n > 0) {
        CR_CHECK(cudaGraphGetNodes(g, nodes.data(), &n));
    }
    std::vector<cudaGraphNode_t> out;
    for (size_t i = 0; i < n; i++) {
        cudaGraphNodeType t;
        CR_CHECK(cudaGraphNodeGetType(nodes[i], &t));
        if (t == cudaGraphNodeTypeKernel) {
            out.push_back(nodes[i]);
        }
    }
    return out;
}

// set the window on every kernel node of a cudaGraph_t, returns the node count touched
static int64_t apply_window_to_graph(int64_t graph, int64_t ptr, int64_t nbytes, double hit_ratio, int64_t hit_prop, int64_t miss_prop) {
    cudaKernelNodeAttrValue v;
    memset(&v, 0, sizeof(v));
    v.accessPolicyWindow = make_window(ptr, nbytes, hit_ratio, hit_prop, miss_prop);
    std::vector<cudaGraphNode_t> nodes = kernel_nodes(graph);
    for (cudaGraphNode_t node : nodes) {
        CR_CHECK(cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, &v));
    }
    return (int64_t) nodes.size();
}

// first kernel node window as ([base_ptr, num_bytes, hit_prop, miss_prop], hit_ratio), or None without kernel nodes
static py::object get_graph_window(int64_t graph) {
    std::vector<cudaGraphNode_t> nodes = kernel_nodes(graph);
    if (nodes.empty()) {
        return py::none();
    }
    cudaKernelNodeAttrValue v;
    memset(&v, 0, sizeof(v));
    CR_CHECK(cudaGraphKernelNodeGetAttribute(nodes[0], cudaKernelNodeAttributeAccessPolicyWindow, &v));
    const cudaAccessPolicyWindow & w = v.accessPolicyWindow;
    std::vector<int64_t> out = {(int64_t) (intptr_t) w.base_ptr, (int64_t) w.num_bytes, (int64_t) w.hitProp, (int64_t) w.missProp};
    return py::make_tuple(out, (double) w.hitRatio);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_persisting_bytes", &max_persisting_bytes);
    m.def("max_window_bytes", &max_window_bytes);
    m.def("l2_bytes", &l2_bytes);
    m.def("get_limit", &get_limit);
    m.def("set_limit", &set_limit);
    m.def("reset", &reset);
    m.def("set_stream_window", &set_stream_window);
    m.def("get_stream_window", &get_stream_window);
    m.def("apply_window_to_graph", &apply_window_to_graph);
    m.def("get_graph_window", &get_graph_window);
}
