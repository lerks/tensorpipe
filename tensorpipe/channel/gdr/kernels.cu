/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <tensorpipe/channel/gdr/kernels.h>

#include <cassert>

#include <tensorpipe/channel/gdr/common.h>

namespace tensorpipe {
namespace channel {
namespace gdr {

// CUDA 10.2 introduced libcu++ (a STL for CUDA) which has cuda::std::atomic
// (STL-compliant) and cuda::atomic (with extensions) which might help us here.
// However I couldn't find docs for libcu++ and, anyways, we need to support
// older versions of CUDA too.

__global__ void sendKernel(SendKernelArgs args) {
  // This kernel copies the contents of a source buffer (user-provided, of
  // arbitrary size) to a staging ringbuffer (allocated by the channel,
  // registered with the InfiniBand NIC). Both of them reside on the GPU.
  // FIXME Should this data also be volatile? (See below)
  const uint8_t* sourceBuff = args.sourceBuff;
  const size_t sourceBuffSize = args.sourceBuffSize;
  uint8_t* stagingBuff = args.stagingBuff;
  const size_t stepSize = args.stagingBuffSize / kPipelineParallelism;

  // Pointers to the head and tail of the ringbuffer, located on mapped and
  // page-locked host memory. Qualified as volatile to inform the compiler that
  // these memory locations are read/written by others and thus access to them
  // should be left unoptimized and uncached.
  // We do _not_ need to operate on them using atomic primitives, first of all
  // because atomics don't work on mapped host memory but also because the CUDA
  // runtime already guarantees that aligned loads and stores to data of up to
  // 8 bytes is performed in a single access.
  volatile uint64_t* sharedHeadPtr = args.sharedHeadPtr;
  const volatile uint64_t* sharedTailPtr = args.sharedTailPtr;

  // Private on-device copies of the values of the above pointers. However,
  // since those pointers hold the "source of truth" for head/tail, these values
  // are just caches, for performance.
  uint64_t deviceHead = *sharedHeadPtr;
  uint64_t deviceTail = *sharedTailPtr;

  // Successive transfers don't overlap (yet?) so when this one starts the
  // previous one has fully completed and the ringbuffer is therefore empty.
  assert(deviceHead == args.start);
  assert(deviceTail == args.start);

  for (uint64_t step = args.start; step < args.end; step++) {
    const uint8_t* src = sourceBuff + step * stepSize;
    uint8_t* dst = stagingBuff + (step % kPipelineParallelism) * stepSize;
    size_t count = min(stepSize, sourceBuffSize - step * stepSize);

    // We'll produce the chunk at the head, but can only do so when head - tail
    // < kPipelineParallelism.
    // This should be an atomic operation with an "acquire" memory order. Since
    // CUDA doesn't have a primitive for that, we work around it through a full
    // memory barrier which, in particular, means that no read/write scheduled
    // after the fence can be reordered and performed before the fence.
    while (deviceTail + kPipelineParallelism <= deviceHead) {
      deviceTail = *sharedTailPtr;
    }
    __threadfence_system();

    // This is a single-threaded linear memcpy, which is probably awfully slow.
    memcpy(dst, src, count);

    // Notify the host that we've produced the chunk at the tail and it can go
    // ahead and send it over InfiniBand.
    // This should be an atomic operation with a "release" memory order. Since
    // CUDA doesn't have a primitive for that, we work around it through a full
    // memory barrier which, in particular, means that no read/write scheduled
    // before the fence can be reordered and performed after the fence.
    __threadfence_system();
    deviceHead += 1;
    *sharedHeadPtr = deviceHead;
  }
}

__global__ void recvKernel(RecvKernelArgs args) {
  // This kernel copies the contents of a staging ringbuffer (allocated by the
  // channel, registered with the InfiniBand NIC) to a target buffer
  // (user-provided, of arbitrary size). Both of them reside on the GPU.
  // FIXME Should this data also be volatile? (See below)
  uint8_t* targetBuff = args.targetBuff;
  const size_t targetBuffSize = args.targetBuffSize;
  const uint8_t* stagingBuff = args.stagingBuff;
  const size_t stepSize = args.stagingBuffSize / kPipelineParallelism;

  // Pointers to the head and tail of the ringbuffer, located on mapped and
  // page-locked host memory. Qualified as volatile to inform the compiler that
  // these memory locations are read/written by others and thus access to them
  // should be left unoptimized and uncached.
  // We do _not_ need to operate on them using atomic primitives, first of all
  // because atomics don't work on mapped host memory but also because the CUDA
  // runtime already guarantees that aligned loads and stores to data of up to
  // 8 bytes is performed in a single access.
  const volatile uint64_t* sharedHeadPtr = args.sharedHeadPtr;
  volatile uint64_t* sharedTailPtr = args.sharedTailPtr;

  // Private on-device copies of the values of the above pointers. However,
  // since those pointers hold the "source of truth" for head/tail, these values
  // are just caches, for performance.
  uint64_t deviceHead = *sharedHeadPtr;
  uint64_t deviceTail = *sharedTailPtr;

  // Successive transfers don't overlap (yet?) so when this one starts the
  // previous one has fully completed and the ringbuffer is therefore empty.
  assert(deviceHead == args.start);
  assert(deviceTail == args.start);

  for (uint64_t step = args.start; step < args.end; step++) {
    const uint8_t* src = stagingBuff + (step % kPipelineParallelism) * stepSize;
    uint8_t* dst = targetBuff + step * stepSize;
    size_t count = min(stepSize, targetBuffSize - step * stepSize);

    // We'll consume the chunk at the tail, but can only do so when tail < head.
    // This should be an atomic operation with an "acquire" memory order. Since
    // CUDA doesn't have a primitive for that, we work around it through a full
    // memory barrier which, in particular, means that no read/write scheduled
    // after the fence can be reordered and performed before the fence.
    while (deviceHead <= deviceTail) {
      deviceHead = *sharedHeadPtr;
    }
    __threadfence_system();

    // This is a single-threaded linear memcpy, which is probably awfully slow.
    memcpy(dst, src, count);

    // Notify the host that we've produced the chunk at the tail and it can go
    // ahead and send it over InfiniBand.
    // This should be an atomic operation with a "release" memory order. Since
    // CUDA doesn't have a primitive for that, we work around it through a full
    // memory barrier which, in particular, means that no read/write scheduled
    // before the fence can be reordered and performed after the fence.
    __threadfence_system();
    deviceTail += 1;
    *sharedTailPtr = deviceTail;
  }
}

} // namespace gdr
} // namespace channel
} // namespace tensorpipe