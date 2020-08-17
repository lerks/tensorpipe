/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace tensorpipe {
namespace channel {
namespace gdr {

static constexpr auto kStagingBufferSize = 4 * 1024 * 1024; // 4 MiB
static constexpr auto kPipelineParallelism = 8;

// For now, use only one IB device, only one GPU, only one port, only one NUMA
// node, ...

static constexpr auto kGpuIdx = 0;
static constexpr auto kIbvHca = "mlx5_0";
static constexpr auto kIbvPort = 1;
static constexpr auto kNumaNode = 1;

} // namespace gdr
} // namespace channel
} // namespace tensorpipe
