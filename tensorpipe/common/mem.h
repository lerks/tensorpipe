/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstring>

#include <cuda_runtime.h>

#include <tensorpipe/common/defs.h>

inline size_t getPageSize() {
  static size_t pageSize = sysconf(_SC_PAGESIZE);
  return pageSize;
}

class PageAlignedPtr {
 public:
  PageAlignedPtr() = default;

  explicit PageAlignedPtr(size_t size) {
    size_t pageSize = getPageSize();
    // TP_THROW_ASSERT_IF(size % pageSize != 0);
    void* ptr;
    int rv = posix_memalign(&ptr, pageSize, size);
    TP_THROW_SYSTEM_IF(rv != 0, rv);
    std::memset(ptr, 0, size);
    ptr_ = decltype(ptr_)(reinterpret_cast<uint8_t*>(ptr));
  }

  uint8_t* ptr() {
    return ptr_.get();
  }

  void reset() {
    ptr_.reset();
  }

 private:
  struct Deleter {
    void operator()(void* ptr) {
      free(ptr);
    }
  };

  std::unique_ptr<uint8_t[], Deleter> ptr_;
};

class CudaDevicePtr {
 public:
  CudaDevicePtr() = default;

  explicit CudaDevicePtr(size_t size) {
    void* ptr;
    TP_CHECK_CUDA(cudaMalloc(&ptr, size));
    TP_CHECK_CUDA(cudaMemset(ptr, 0, size));
    ptr_ = decltype(ptr_)(reinterpret_cast<uint8_t*>(ptr));
  }

  uint8_t* ptr() {
    return ptr_.get();
  }

  void reset() {
    ptr_.reset();
  }

 private:
  struct Deleter {
    void operator()(void* ptr) {
      TP_CHECK_CUDA(cudaFree(ptr));
    }
  };

  std::unique_ptr<uint8_t[], Deleter> ptr_;
};

class CudaHostPtr {
 public:
  CudaHostPtr() = default;

  explicit CudaHostPtr(size_t size) {
    void* ptr;
    TP_CHECK_CUDA(cudaHostAlloc(&ptr, size, cudaHostAllocMapped));
    std::memset(ptr, 0, size);
    ptr_ = decltype(ptr_)(reinterpret_cast<uint8_t*>(ptr));
  }

  uint8_t* ptr() {
    return ptr_.get();
  }

  void reset() {
    ptr_.reset();
  }

 private:
  struct Deleter {
    void operator()(void* ptr) {
      TP_CHECK_CUDA(cudaFreeHost(ptr));
    }
  };

  std::unique_ptr<uint8_t[], Deleter> ptr_;
};
