/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <list>
#include <memory>
#include <string>

#include <cuda_runtime.h>

#include <tensorpipe/channel/channel_impl_boilerplate.h>
#include <tensorpipe/common/cuda.h>
#include <tensorpipe/common/cuda_buffer.h>
#include <tensorpipe/common/cuda_lib.h>
#include <tensorpipe/common/state_machine.h>
#include <tensorpipe/transport/context.h>

namespace tensorpipe {
namespace channel {
namespace cuda_ipc {

class ContextImpl;

struct SendOperation {
  enum State { UNINITIALIZED, READING_REPLY, FINISHED };

  // Fields used by the state machine
  uint64_t sequenceNumber{0};
  State state{UNINITIALIZED};

  // Progress flags
  bool doneReadingReply{false};

  // Arguments at creation
  const void* const ptr;
  const int deviceIdx;
  const cudaStream_t stream;
  TDescriptorCallback descriptorCallback;
  TSendCallback callback;

  // Other data
  CudaEvent startEv;
  std::string stopEvHandle;

  SendOperation(
      TDescriptorCallback descriptorCallback,
      TSendCallback callback,
      int deviceIdx,
      const void* ptr,
      cudaStream_t stream);
};

struct RecvOperation {
  enum State { UNINITIALIZED, READING_ACK, FINISHED };

  // Fields used by the state machine
  uint64_t sequenceNumber{0};
  State state{UNINITIALIZED};

  // Progress flags
  bool doneReadingAck{false};

  // Arguments at creation
  void* const ptr;
  const size_t length;
  const int deviceIdx;
  const cudaStream_t stream;
  TRecvCallback callback;

  // Other data
  CudaEvent stopEv;
  std::string allocationId;
  std::string bufferHandle;
  size_t offset;
  std::string startEvHandle;

  RecvOperation(int deviceIdx, void* ptr, cudaStream_t stream, size_t length);
};

class ChannelImpl final
    : public ChannelImplBoilerplate<CudaBuffer, ContextImpl, ChannelImpl> {
 public:
  ChannelImpl(
      ConstructorToken token,
      std::shared_ptr<ContextImpl> context,
      std::string id,
      std::shared_ptr<transport::Connection> replyConnection,
      std::shared_ptr<transport::Connection> ackConnection);

 protected:
  // Implement the entry points called by ChannelImplBoilerplate.
  void initImplFromLoop() override;
  void sendImplFromLoop(
      uint64_t sequenceNumber,
      CudaBuffer buffer,
      TDescriptorCallback descriptorCallback,
      TSendCallback callback) override;
  void recvImplFromLoop(
      uint64_t sequenceNumber,
      TDescriptor descriptor,
      CudaBuffer buffer,
      TRecvCallback callback) override;
  void handleErrorImpl() override;

 private:
  const std::shared_ptr<transport::Connection> replyConnection_;
  const std::shared_ptr<transport::Connection> ackConnection_;

  OpsStateMachine<ChannelImpl, SendOperation> sendOps_{
      *this,
      &ChannelImpl::advanceSendOperation};
  using SendOpIter = decltype(sendOps_)::Iter;
  OpsStateMachine<ChannelImpl, RecvOperation> recvOps_{
      *this,
      &ChannelImpl::advanceRecvOperation};
  using RecvOpIter = decltype(recvOps_)::Iter;

  // State machines for send and recv ops.
  void advanceSendOperation(
      SendOpIter opIter,
      SendOperation::State prevOpState);
  void advanceRecvOperation(
      RecvOpIter opIter,
      RecvOperation::State prevOpState);

  // Actions (i.e., methods that begin a state transition).
  // For send operations:
  void recordStartEvent(SendOpIter opIter);
  void callDescriptorCallback(SendOpIter opIter);
  void readReply(SendOpIter opIter);
  void waitOnStopEvent(SendOpIter opIter);
  void callSendCallback(SendOpIter opIter);
  void writeAck(SendOpIter opIter);
  // For recv operations:
  void waitOnStartEventAndCopyAndRecordStopEvent(RecvOpIter opIter);
  void callRecvCallback(RecvOpIter opIter);
  void writeReplyAndReadAck(RecvOpIter opIter);
};

} // namespace cuda_ipc
} // namespace channel
} // namespace tensorpipe
