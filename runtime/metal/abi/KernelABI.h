#pragma once

// Shader prelude. The ABI headers below also compile as host C++; their
// shared parameter layouts use fixed-width scalars and explicit padding.

#include "metal/abi/DraftAttention.h"
#include "metal/abi/Embedding.h"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/GDN.h"
#include "metal/abi/Linear.h"
#include "metal/abi/MoE.h"
#include "metal/abi/PagedAttention.h"
#include "metal/abi/RoPE.h"
#include "metal/abi/RowCopy.h"
#include "metal/abi/Sampling.h"
#include "metal/abi/Vision.h"
#include <metal_stdlib>
// MetalPerformancePrimitives is empty below Metal 4.0 (macOS 15 build,
// Makefile MACOS15=1, compiles Apple7/8 kernels at -std=metal3.2): only the
// Apple9+ kernels that actually use mpp::tensor_ops need it.
#if __METAL_VERSION__ >= 400
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

using namespace metal;
#if __METAL_VERSION__ >= 400
using namespace mpp::tensor_ops;
#endif
