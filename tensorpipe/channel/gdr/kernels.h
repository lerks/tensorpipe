/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define CUDA_GLOBAL __global__
#else
#define CUDA_GLOBAL
#endif

namespace tensorpipe {
namespace channel {
namespace gdr {

struct SendKernelArgs {
  const uint8_t* sourceBuff;
  size_t sourceBuffSize;

  uint8_t* stagingBuff;
  size_t stagingBuffSize;

  uint64_t start;
  uint64_t end;

  volatile uint64_t* sharedHeadPtr;
  const volatile uint64_t* sharedTailPtr;
};

void CUDA_GLOBAL sendKernel(SendKernelArgs args);

struct RecvKernelArgs {
  uint8_t* targetBuff;
  size_t targetBuffSize;

  const uint8_t* stagingBuff;
  size_t stagingBuffSize;

  uint64_t start;
  uint64_t end;

  const volatile uint64_t* sharedHeadPtr;
  volatile uint64_t* sharedTailPtr;
};

void CUDA_GLOBAL recvKernel(RecvKernelArgs args);

} // namespace gdr
} // namespace channel
} // namespace tensorpipe
