/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <tensorpipe/channel/gdr/channel.h>

#include <algorithm>
#include <list>

#include <cuda_runtime.h>

#include <tensorpipe/channel/error.h>
#include <tensorpipe/channel/helpers.h>
#include <tensorpipe/channel/gdr/common.h>
#include <tensorpipe/channel/gdr/kernels.h>
#include <tensorpipe/common/callback.h>
#include <tensorpipe/common/defs.h>
#include <tensorpipe/common/error.h>
#include <tensorpipe/common/error_macros.h>
#include <tensorpipe/common/ibv.h>
#include <tensorpipe/common/mem.h>
#include <tensorpipe/proto/channel/gdr.pb.h>

namespace tensorpipe {
namespace channel {
namespace gdr {

namespace {

uint64_t divideAndRoundUp(uint64_t value, uint64_t factor) {
  TP_THROW_ASSERT_IF(factor == 0);
  if (unlikely(value == 0)) {
    return 0;
  }
  return (value - 1) / factor + 1;
}

uint64_t roundUpToNearestMultiple(uint64_t value, uint64_t factor) {
  TP_THROW_ASSERT_IF(factor == 0);
  if (unlikely(value == 0)) {
    return 0;
  }
  return divideAndRoundUp(value, factor) * factor;
}

struct SendOperation {
  const uint8_t* ptr;
  size_t length;

  Channel::TSendCallback callback;
  //
  // TODO Cuda stream?

  uint64_t start;
  uint64_t end;
};

struct RecvOperation {
  uint8_t* ptr;
  size_t length;

  Channel::TSendCallback callback;
  //
  // TODO Cuda stream?

  uint64_t start;
  uint64_t end;
};

IbvQueuePair createIbvQueuePair(
    IbvProtectionDomain& pd,
    IbvCompletionQueue& cq) {
  struct ibv_qp_init_attr initAttr;
  std::memset(&initAttr, 0, sizeof(initAttr));
  initAttr.qp_type = IBV_QPT_RC;
  initAttr.send_cq = cq.ptr();
  initAttr.recv_cq = cq.ptr();
  initAttr.cap.max_send_wr = 2 * kPipelineParallelism;
  initAttr.cap.max_recv_wr = kPipelineParallelism;
  initAttr.cap.max_send_sge = 1;
  initAttr.cap.max_recv_sge = 1;
  initAttr.sq_sig_all = 1;
  return IbvQueuePair(pd, initAttr);
}

} // namespace

class Channel::Impl : public std::enable_shared_from_this<Channel::Impl> {
 public:
  Impl(
      std::shared_ptr<Context::PrivateIface>,
      std::shared_ptr<transport::Connection>,
      std::string);

  // Called by the channel's constructor.
  void init();

  void send(
      const void* ptr,
      size_t length,
      TDescriptorCallback descriptorCallback,
      TSendCallback callback);

  void recv(
      TDescriptor descriptor,
      void* ptr,
      size_t length,
      TRecvCallback callback);

  // Tell the channel what its identifier is.
  void setId(std::string id);

  void close();

 private:
  OnDemandLoop loop_;

  void sendFromLoop_(
      const void* ptr,
      size_t length,
      TDescriptorCallback descriptorCallback,
      TSendCallback callback);

  void recvFromLoop_(
      TDescriptor descriptor,
      void* ptr,
      size_t length,
      TRecvCallback callback);

  void setIdFromLoop_(std::string id);

  void initFromLoop_();

  void finishInitFromLoop_(const proto::IbvSetupInfo& info);

  void closeFromLoop_();

  void setError_(Error error);

  // Helper function to process transport error.
  // Shared between read and write callback entry points.
  void handleError_();

  std::shared_ptr<Context::PrivateIface> context_;
  std::shared_ptr<transport::Connection> connection_;
  Error error_{Error::kSuccess};
  ClosingReceiver closingReceiver_;

  // Increasing identifier for send operations.
  uint64_t nextTensorBeingSent_{0};

  // Increasing identifier for recv operations.
  uint64_t nextTensorBeingReceived_{0};

  // An identifier for the channel, composed of the identifier for the context,
  // combined with an increasing sequence number. It will only be used for
  // logging and debugging purposes.
  std::string id_;

  // Helpers to prepare callbacks from transports
  LazyCallbackWrapper<Impl> lazyCallbackWrapper_{*this, this->loop_};
  EagerCallbackWrapper<Impl> eagerCallbackWrapper_{*this, this->loop_};

  // For some odd reason it seems we need to use a qualified name here...
  template <typename T>
  friend class tensorpipe::LazyCallbackWrapper;
  template <typename T>
  friend class tensorpipe::EagerCallbackWrapper;

  enum class State { INITIALIZING, WAITING_FOR_IBV_INFO, READY };

  State state_{State::INITIALIZING};

  std::list<SendOperation> sendOps_;
  std::list<RecvOperation> recvOps_;

  // IB control structures.
  struct {
    IbvProtectionDomain pd;
    IbvCompletionQueue cq;
    IbvQueuePair qp;

    IbvSetupInformation selfInfo_;

    // We set the MTU to the minimum of the active MTUs of the ports of the two
    // endpoints. Thus we need it twice: to send to the other side, and then to
    // take the min. In this field we cache the value so we onky query it once.
    enum ibv_mtu mtu;
  } ibv;

  // The outbox and inbox are staging buffers that are used to exchange data
  // between the GPU and the NIC. They reside on the GPU but are registered with
  // the NIC, which can thus access them on its own through PCI peer-to-peer.
  // The control structures associated with these buffers must be accessed by
  // both host and device and are thus allocated on page-locked host memory, so
  // that the GPU can also access them on its own, again through PCI P2P. For
  // performance reasons, both the host and the device have a private copy of
  // these control structures, which they use as a cache. These staging buffers
  // operate as ring buffers, split into chunks (see kPipelineParallelism). Each
  // step handles a whole chunk. The chunks in use (those that contain data in
  // flight) are [tail, head).

  //                ..................current op..................
  // +--------------+--------------+--------------+--------------+----------- -
  // +--------------+--------------+--------------+--------------+----------- -
  // 0              start          tail           head           end

  // The outbox is used for data that is being sent to the remote endpoint. It
  // contains the chunks that have already been copied from the source buffer by
  // the CUDA kernel and that are currently being sent by InfiniBand (through a
  // RDMA write). Thus:
  // - the GPU (the device) increases the head;
  // - the NIC (the host) increases the tail.
  struct {
    // The on-device staging area. The CUDA kernel copies to it from the source
    // CUDA buffer, and then IB then reads from it.
    CudaDevicePtr stagingBuffer;
    IbvMemoryRegion stagingBufferMr;

    // The pointer of the inbox of the remote end (it is relative to their
    // address space, and has no meaning on this side), and the remote key used
    // by InfiniBand to identify the inbox's memory region.
    uint64_t remoteStagingBufferPtr{0};
    uint32_t remoteStagingBufferRkey{0};

    // The control flags for the CUDA kernel to synchronize with the host.
    // Allocated on host memory but page-locked and mapped to device too.
    // Two separate allocations, each one page in size, one for each "direction"
    // of communication (between host and device) in order for each to have only
    // one producer and one consumer.
    CudaHostPtr deviceToHost; // Stores the head.
    CudaHostPtr hostToDevice; // Stores the tail.

    // Pointers to the above allocations, for easier access. Qualified as
    // volatile to tell the compiler that operations on them have "side effects"
    // (interact with other users) and thus cannot be optimizied or cached.
    // They are also wrapped in atomic helpers, for two reason:
    // - For the loads and stores to operate on consistent snapshots, and not
    //   access a corrupted value when they race with other operations. For
    //   this, a "relaxed" memory order would be enough.
    // - So that operations on these control structures are correctly ordered
    //   against operations on the data. For example, when we write some data
    //   and then increase the head, the compiler may reorder these writes,
    //   causing the device to see the increased head and access the data before
    //   it has really been updated. By using the "release" and "acquire" memory
    //   orders we prevent the compiler from doing these "optimizations".
    const volatile std::atomic<uint64_t>* sharedHeadPtr{nullptr};
    volatile std::atomic<uint64_t>* sharedTailPtr{nullptr};
    static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t), "!");

    // The host's private copies of the above shared values. The shared ones are
    // the source of truth, so these are just "cached" values.
    uint64_t hostHead{0};
    uint64_t hostTail{0};
  } outbox;

  // The inbox is used for data that is being received from the remote endpoint.
  // It contains the chunks that have already been received by InfiniBand
  // (through a RDMA write performed by the remote) and that are currently being
  // copied to the target buffer by the CUDA kernel. Thus:
  // - the NIC (the host) increases the head;
  // - the GPU (the device) increases the tail.
  struct {
    // The on-device staging area. IB writes to it, and then the CUDA kernel
    // copies from it to the target CUDA buffer.
    CudaDevicePtr stagingBuffer;
    IbvMemoryRegion stagingBufferMr;

    // The control flags for the CUDA kernel to synchronize with the host.
    // Allocated on host memory but page-locked and mapped to device too.
    // Two separate allocations, each one page in size, one for each "direction"
    // of communication (between host and device) in order for each to have only
    // one producer and one consumer.
    CudaHostPtr hostToDevice; // Stores the head.
    CudaHostPtr deviceToHost; // Stores the tail.

    // Pointers to the above allocations, for easier access. Qualified as
    // volatile to tell the compiler that operations on them have "side effects"
    // (interact with other users) and thus cannot be optimizied or cached.
    // They are also wrapped in atomic helpers, for two reason:
    // - For the loads and stores to operate on consistent snapshots, and not
    //   access a corrupted value when they race with other operations. For
    //   this, a "relaxed" memory order would be enough.
    // - So that operations on these control structures are correctly ordered
    //   against operations on the data. For example, when we write some data
    //   and then increase the head, the compiler may reorder these writes,
    //   causing the device to see the increased head and access the data before
    //   it has really been updated. By using the "release" and "acquire" memory
    //   orders we prevent the compiler from doing these "optimizations".
    volatile std::atomic<uint64_t>* sharedHeadPtr{nullptr};
    const volatile std::atomic<uint64_t>* sharedTailPtr{nullptr};
    static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t), "!");

    // The host's private copies of the above shared values. The shared ones are
    // the source of truth, so these are just "cached" values.
    uint64_t hostHead{0};
    uint64_t hostTail{0};
  } inbox;

  // FIXME make these part of inbox/outbox.
  uint64_t totalSendSteps{0};
  uint64_t totalRecvSteps{0};

  void sendOneOperation(SendOperation& sendOp);
  void receiveOneOperation(RecvOperation& recvOp);

  // Called when the GPU has copied one chunk into the outbox, in order to send
  // it over IB.
  void performSend();
  // Called when we receive the confirmation that the send has been received,
  // freeing up a chunk in the ouboux, allowing the GPU to copy a new one in.
  void acknowledgeSend();

  // Called when we receive a notification that the remote end has written one
  // chunk into the inbox, to kick off the GPU to copy it into the target.
  void handleRecv();
  // Called when the GPU has finished copying one chunk from the inbox, so that
  // we can inform the sender that they can write the next one.
  void confirmRecv();

  // Check whether any events are coming in from the InfiniBand NIC.
  bool pollIbv();
  // Check whether we can advance any current ongoing transfer.
  bool progress();
};

Channel::Channel(
    ConstructorToken /* unused */,
    std::shared_ptr<Context::PrivateIface> context,
    std::shared_ptr<transport::Connection> connection,
    std::string id)
    : impl_(std::make_shared<Impl>(
          std::move(context),
          std::move(connection),
          std::move(id))) {
  impl_->init();
}

Channel::Impl::Impl(
    std::shared_ptr<Context::PrivateIface> context,
    std::shared_ptr<transport::Connection> connection,
    std::string id)
    : context_(std::move(context)),
      connection_(std::move(connection)),
      closingReceiver_(context_, context_->getClosingEmitter()),
      id_(std::move(id)) {}

void Channel::send(
    const void* ptr,
    size_t length,
    TDescriptorCallback descriptorCallback,
    TSendCallback callback) {
  impl_->send(ptr, length, std::move(descriptorCallback), std::move(callback));
}

void Channel::Impl::send(
    const void* ptr,
    size_t length,
    TDescriptorCallback descriptorCallback,
    TSendCallback callback) {
  loop_.deferToLoop([this,
                     ptr,
                     length,
                     descriptorCallback{std::move(descriptorCallback)},
                     callback{std::move(callback)}]() mutable {
    sendFromLoop_(
        ptr, length, std::move(descriptorCallback), std::move(callback));
  });
}

// Send memory region to peer.
void Channel::Impl::sendFromLoop_(
    const void* ptr,
    size_t length,
    TDescriptorCallback descriptorCallback,
    TSendCallback callback) {
  TP_DCHECK(loop_.inLoop());

  const uint64_t sequenceNumber = nextTensorBeingSent_++;
  TP_VLOG(4) << "Channel " << id_ << " received a send request (#"
             << sequenceNumber << ")";

  descriptorCallback = [this,
                        sequenceNumber,
                        descriptorCallback{std::move(descriptorCallback)}](
                           const Error& error, TDescriptor descriptor) {
    // There is no requirement for the channel to invoke callbacks in order.
    TP_VLOG(4) << "Channel " << id_ << " is calling a descriptor callback (#"
               << sequenceNumber << ")";
    descriptorCallback(error, std::move(descriptor));
    TP_VLOG(4) << "Channel " << id_ << " done calling a descriptor callback (#"
               << sequenceNumber << ")";
  };

  callback = [this, sequenceNumber, callback{std::move(callback)}](
                 const Error& error) {
    // There is no requirement for the channel to invoke callbacks in order.
    TP_VLOG(4) << "Channel " << id_ << " is calling a send callback (#"
               << sequenceNumber << ")";
    callback(error);
    TP_VLOG(4) << "Channel " << id_ << " done calling a send callback (#"
               << sequenceNumber << ")";
  };

  if (error_) {
    descriptorCallback(error_, std::string());
    callback(error_);
    return;
  }

  descriptorCallback(Error::kSuccess, std::string());

  SendOperation sendOp;
  sendOp.ptr = reinterpret_cast<const uint8_t*>(ptr);
  sendOp.length = length;
  sendOp.callback = std::move(callback);
  sendOps_.push_back(std::move(sendOp));

  if (state_ == State::READY) {
    sendOneOperation(sendOps_.back());
  }
}

// Receive memory region from peer.
void Channel::recv(
    TDescriptor descriptor,
    void* ptr,
    size_t length,
    TRecvCallback callback) {
  impl_->recv(std::move(descriptor), ptr, length, std::move(callback));
}

void Channel::Impl::recv(
    TDescriptor descriptor,
    void* ptr,
    size_t length,
    TRecvCallback callback) {
  loop_.deferToLoop([this,
                     descriptor{std::move(descriptor)},
                     ptr,
                     length,
                     callback{std::move(callback)}]() mutable {
    recvFromLoop_(std::move(descriptor), ptr, length, std::move(callback));
  });
}

void Channel::Impl::recvFromLoop_(
    TDescriptor descriptor,
    void* ptr,
    size_t length,
    TRecvCallback callback) {
  TP_DCHECK(loop_.inLoop());

  const uint64_t sequenceNumber = nextTensorBeingReceived_++;
  TP_VLOG(4) << "Channel " << id_ << " received a recv request (#"
             << sequenceNumber << ")";

  callback = [this, sequenceNumber, callback{std::move(callback)}](
                 const Error& error) {
    // There is no requirement for the channel to invoke callbacks in order.
    TP_VLOG(4) << "Channel " << id_ << " is calling a recv callback (#"
               << sequenceNumber << ")";
    callback(error);
    TP_VLOG(4) << "Channel " << id_ << " done calling a recv callback (#"
               << sequenceNumber << ")";
  };

  if (error_) {
    callback(error_);
    return;
  }

  TP_DCHECK_EQ(descriptor, std::string());

  RecvOperation recvOp;
  recvOp.ptr = reinterpret_cast<uint8_t*>(ptr);
  recvOp.length = length;
  recvOp.callback = std::move(callback);
  recvOps_.push_back(std::move(recvOp));

  if (state_ == State::READY) {
    receiveOneOperation(recvOps_.back());
  }
}

void Channel::Impl::init() {
  loop_.deferToLoop([this]() { initFromLoop_(); });
}

void Channel::Impl::initFromLoop_() {
  TP_DCHECK(loop_.inLoop());

  closingReceiver_.activate(*this);

  IbvContext& ibvCtx = context_->getIbvContext();
  this->ibv.pd = IbvProtectionDomain(ibvCtx);
  // Size it for up to four times the parallelism factor as for each chunk there
  // could be four types of events: two for sending and two for receiving, the
  // completion of the outgoing operation and the notification of the incoming
  // one.
  this->ibv.cq = IbvCompletionQueue(
      ibvCtx,
      /*cqe=*/4 * kPipelineParallelism,
      /*cq_context=*/NULL,
      /*channel=*/NULL,
      /*comp_vector=*/0);
  this->ibv.qp = createIbvQueuePair(this->ibv.pd, this->ibv.cq);
  transitionIbvQueuePairToInit(this->ibv.qp, context_->getIbvAddress());

  // The outbox is only used as the source for RDMA writes, so it doesn't need
  // any remote access flags.
  this->outbox.stagingBuffer = CudaDevicePtr(kStagingBufferSize);
  this->outbox.stagingBufferMr = IbvMemoryRegion(
      this->ibv.pd,
      this->outbox.stagingBuffer.ptr(),
      kStagingBufferSize,
      /*accessFlags=*/0);

  this->outbox.hostToDevice =
      CudaHostPtr(roundUpToNearestMultiple(sizeof(uint64_t), getPageSize()));
  this->outbox.deviceToHost =
      CudaHostPtr(roundUpToNearestMultiple(sizeof(uint64_t), getPageSize()));

  this->outbox.sharedHeadPtr = new(this->outbox.deviceToHost.ptr()) std::atomic<uint64_t>(0);
  this->outbox.sharedTailPtr = new(this->outbox.hostToDevice.ptr()) std::atomic<uint64_t>(0);
  TP_THROW_ASSERT_IF(reinterpret_cast<void*>(&this->outbox.sharedHeadPtr) != reinterpret_cast<void*>(this->outbox.deviceToHost.ptr()));
  TP_THROW_ASSERT_IF(!this->outbox.sharedHeadPtr->is_lock_free());
  TP_THROW_ASSERT_IF(reinterpret_cast<void*>(&this->outbox.sharedTailPtr) != reinterpret_cast<void*>(this->outbox.hostToDevice.ptr()));
  TP_THROW_ASSERT_IF(!this->outbox.sharedTailPtr->is_lock_free());

  this->inbox.stagingBuffer = CudaDevicePtr(kStagingBufferSize);
  this->inbox.stagingBufferMr = IbvMemoryRegion(
      this->ibv.pd,
      this->inbox.stagingBuffer.ptr(),
      kStagingBufferSize,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);

  this->inbox.deviceToHost =
      CudaHostPtr(roundUpToNearestMultiple(sizeof(uint64_t), getPageSize()));
  this->inbox.hostToDevice =
      CudaHostPtr(roundUpToNearestMultiple(sizeof(uint64_t), getPageSize()));

  this->inbox.sharedHeadPtr = new(this->inbox.hostToDevice.ptr()) std::atomic<uint64_t>(0);
  this->inbox.sharedTailPtr = new(this->inbox.deviceToHost.ptr()) std::atomic<uint64_t>(0);
  TP_THROW_ASSERT_IF(reinterpret_cast<void*>(&this->inbox.sharedHeadPtr) != reinterpret_cast<void*>(this->inbox.hostToDevice.ptr()));
  TP_THROW_ASSERT_IF(!this->inbox.sharedHeadPtr->is_lock_free());
  TP_THROW_ASSERT_IF(reinterpret_cast<void*>(&this->inbox.sharedTailPtr) != reinterpret_cast<void*>(this->inbox.deviceToHost.ptr()));
  TP_THROW_ASSERT_IF(!this->inbox.sharedTailPtr->is_lock_free());

  auto pbPacketOut = std::make_shared<proto::Packet>();
  proto::IbvSetupInfo* pbInfo = pbPacketOut->mutable_ibv_setup_info();

  ibv.selfInfo_ =
      getIbvSetupInformation(context_->getIbvAddress(), this->ibv.qp);

  pbInfo->set_local_identifier(ibv.selfInfo_.localIdentifier);
  pbInfo->set_subnet_prefix(
      ibv.selfInfo_.globalIdentifier.global.subnet_prefix);
  pbInfo->set_interface_identifier(
      ibv.selfInfo_.globalIdentifier.global.interface_id);
  pbInfo->set_queue_pair_number(ibv.selfInfo_.queuePairNumber);
  // pbInfo->set_maximum_transmission_unit(ibvMtuToInt32(portAttr.active_mtu));
  pbInfo->set_staging_buffer_remote_key(this->inbox.stagingBufferMr->rkey);
  pbInfo->set_staging_buffer_ptr(
      reinterpret_cast<uintptr_t>(this->inbox.stagingBuffer.ptr()));

  connection_->write(
      *pbPacketOut, lazyCallbackWrapper_([pbPacketOut](Impl& impl) {
        TP_VLOG(6) << "Pipe " << impl.id_
                   << " done writing proto (InfiniBand info)";
      }));

  auto pbPacketIn = std::make_shared<proto::Packet>();
  connection_->read(*pbPacketIn, lazyCallbackWrapper_([pbPacketIn](Impl& impl) {
    TP_VLOG(6) << "Pipe " << impl.id_
               << " done reading proto (InfiniBand info)";
    const proto::IbvSetupInfo& pbInfo = pbPacketIn->ibv_setup_info();
    impl.finishInitFromLoop_(pbInfo);
  }));

  state_ = State::WAITING_FOR_IBV_INFO;

  // Store this as we'll need it when we receive the peer's descriptor.
  // this->ibv.mtu = portAttr.active_mtu;
}

void Channel::Impl::finishInitFromLoop_(const proto::IbvSetupInfo& info) {
  TP_DCHECK(loop_.inLoop());

  IbvSetupInformation destinationInfo;
  destinationInfo.localIdentifier = info.local_identifier();
  destinationInfo.globalIdentifier.global.subnet_prefix = info.subnet_prefix();
  destinationInfo.globalIdentifier.global.interface_id =
      info.interface_identifier();
  destinationInfo.queuePairNumber = info.queue_pair_number();
  destinationInfo.packetSequenceNumber = 0;

  transitionIbvQueuePairToReadyToReceive(
      this->ibv.qp, context_->getIbvAddress(), destinationInfo);
  // std::min(ibvInt32ToMtu(info.maximum_transmission_unit()), this->ibv.mtu)
  transitionIbvQueuePairToReadyToSend(this->ibv.qp, ibv.selfInfo_);

  this->outbox.remoteStagingBufferPtr = info.staging_buffer_ptr();
  this->outbox.remoteStagingBufferRkey = info.staging_buffer_remote_key();

  state_ = State::READY;

  for (auto& sendOp : sendOps_) {
    sendOneOperation(sendOp);
  }
  for (auto& recvOp : recvOps_) {
    receiveOneOperation(recvOp);
  }
}

void Channel::Impl::sendOneOperation(SendOperation& sendOp) {
  TP_DCHECK(loop_.inLoop());

  size_t stepSize = kStagingBufferSize / kPipelineParallelism;
  uint64_t numSteps = divideAndRoundUp(sendOp.length, stepSize);

  // TODO In fact, we could now create a separate object for the rest of the
  // transfer, since it has a different set of fields (adds start and end, ...).
  sendOp.start = this->totalSendSteps;
  sendOp.end = this->totalSendSteps + numSteps;
  this->totalSendSteps += numSteps;

  struct SendKernelArgs args;
  args.sourceBuff = sendOp.ptr;
  args.sourceBuffSize = sendOp.length;
  args.stagingBuff = this->outbox.stagingBuffer.ptr();
  args.stagingBuffSize = kStagingBufferSize;
  args.start = sendOp.start;
  args.end = sendOp.end;
  args.sharedHeadPtr = reinterpret_cast<uint64_t*>(this->outbox.deviceToHost.ptr());
  args.sharedTailPtr = reinterpret_cast<uint64_t*>(this->outbox.hostToDevice.ptr());

  void* argsPtr = &args;

  TP_CHECK_CUDA(cudaLaunchKernel(
      /*func=*/(void*)sendKernel,
      /*gridDim=*/dim3(),
      /*blockDim=*/dim3(),
      /*args=*/&argsPtr,
      /*sharedMem=*/0,
      /*stream=*/0));

  sendOp.callback(Error::kSuccess);
  sendOp.callback = nullptr;
}

void Channel::Impl::receiveOneOperation(RecvOperation& recvOp) {
  TP_DCHECK(loop_.inLoop());

  size_t stepSize = kStagingBufferSize / kPipelineParallelism;
  uint64_t numSteps = divideAndRoundUp(recvOp.length, stepSize);

  // TODO In fact, we could now create a separate object for the rest of the
  // transfer, since it has a different set of fields (adds start and end, ...).
  recvOp.start = this->totalRecvSteps;
  recvOp.end = this->totalRecvSteps + numSteps;
  this->totalRecvSteps += numSteps;

  struct RecvKernelArgs args;
  args.targetBuff = recvOp.ptr;
  args.targetBuffSize = recvOp.length;
  args.stagingBuff = this->inbox.stagingBuffer.ptr();
  args.stagingBuffSize = kStagingBufferSize;
  args.start = recvOp.start;
  args.end = recvOp.end;
  args.sharedHeadPtr = reinterpret_cast<uint64_t*>(this->inbox.hostToDevice.ptr());
  args.sharedTailPtr = reinterpret_cast<uint64_t*>(this->inbox.deviceToHost.ptr());

  void* argsPtr = &args;

  TP_CHECK_CUDA(cudaLaunchKernel(
      /*func=*/(void*)recvKernel,
      /*gridDim=*/dim3(),
      /*blockDim=*/dim3(),
      /*args=*/&argsPtr,
      /*sharedMem=*/0,
      /*stream=*/0));

  recvOp.callback(Error::kSuccess);
  recvOp.callback = nullptr;
}

void Channel::Impl::performSend() {
  TP_THROW_ASSERT_IF(sendOps_.empty());
  SendOperation& sendOp = sendOps_.front();

  TP_THROW_ASSERT_IF(!(this->outbox.hostTail <= this->outbox.hostHead));
  TP_THROW_ASSERT_IF(!(this->outbox.hostHead < sendOp.end));

  uint64_t step = this->outbox.hostHead;

  const size_t stepSize = kStagingBufferSize / kPipelineParallelism;

  struct ibv_sge sge;
  sge.addr = reinterpret_cast<uint64_t>(this->outbox.stagingBuffer.ptr()) +
      step * stepSize;
  sge.length = std::min(stepSize, sendOp.length - step * stepSize);
  sge.lkey = this->outbox.stagingBufferMr->lkey;

  struct ibv_send_wr wr;
  std::memset(&wr, 0, sizeof(wr));
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.imm_data = step;
  wr.wr.rdma.remote_addr = this->outbox.remoteStagingBufferPtr;
  wr.wr.rdma.rkey = this->outbox.remoteStagingBufferRkey;

  struct ibv_send_wr* badWr = nullptr;
  TP_CHECK_IBV_INT(ibv_post_send(this->ibv.qp.ptr(), &wr, &badWr));
  TP_THROW_ASSERT_IF(badWr != nullptr);

  // FIXME Should we do a __sync_synchronize here?
  this->outbox.hostHead += 1;
}

void Channel::Impl::acknowledgeSend() {
  TP_THROW_ASSERT_IF(sendOps_.empty());
  SendOperation& sendOp = sendOps_.front();

  TP_THROW_ASSERT_IF(!(this->outbox.hostTail < this->outbox.hostHead));
  TP_THROW_ASSERT_IF(!(this->outbox.hostHead <= sendOp.end));

  this->outbox.hostTail += 1;
  // By increasing the tail we are taking the ringbuffer chunk that we've just
  // finished sending over InfiniBand and *releasing* control over it by
  // allowing the GPU to access it and write new data into it.
  // The "release" memory order ensures that all reads/writes that were
  // scheduled before increasing the head cannot be reordered and delayed beyond
  // that increase.
  this->outbox.sharedTailPtr->store(this->outbox.hostTail, std::memory_order_release);
  if (this->outbox.hostTail == sendOp.end) {
    sendOps_.pop_front();
  }
}

void Channel::Impl::handleRecv() {
  TP_THROW_ASSERT_IF(recvOps_.empty());
  RecvOperation& recvOp = recvOps_.front();

  TP_THROW_ASSERT_IF(!(this->inbox.hostTail <= this->inbox.hostHead));
  TP_THROW_ASSERT_IF(!(this->inbox.hostHead < recvOp.end));

  // By increasing the head we are taking the ringbuffer chunk that we've just
  // received over Infiniband and *releasing* control over it by allowing the
  // GPU to access it and read data out of it into the target buffer.
  // The "release" memory order ensures that all reads/writes that were
  // scheduled before increasing the head cannot be reordered and delayed beyond
  // that increase.
  this->inbox.hostHead += 1;
  this->inbox.sharedHeadPtr->store(this->inbox.hostHead, std::memory_order_release);
}

void Channel::Impl::confirmRecv() {
  TP_THROW_ASSERT_IF(recvOps_.empty());
  RecvOperation& recvOp = recvOps_.front();

  TP_THROW_ASSERT_IF(!(this->inbox.hostTail < this->inbox.hostHead));
  TP_THROW_ASSERT_IF(!(this->inbox.hostHead <= recvOp.end));

  uint64_t step = this->inbox.hostTail;

  struct ibv_send_wr wr;
  std::memset(&wr, 0, sizeof(wr));
  wr.opcode = IBV_WR_SEND_WITH_IMM;
  wr.imm_data = step;

  struct ibv_send_wr* badWr = nullptr;
  TP_CHECK_IBV_INT(ibv_post_send(this->ibv.qp.ptr(), &wr, &badWr));
  TP_THROW_ASSERT_IF(badWr != nullptr);

  this->inbox.hostTail += 1;
  if (this->inbox.hostTail == recvOp.end) {
    recvOps_.pop_front();
  }
}

bool Channel::Impl::pollIbv() {
  std::array<struct ibv_wc, 4> wcs;
  auto rv = ibv_poll_cq(this->ibv.cq.ptr(), wcs.size(), wcs.data());
  TP_THROW_SYSTEM_IF(rv < 0, errno);

  bool isProgressing = false;
  // FIXME count nuMRecvs
  for (size_t wcIdx = 0; wcIdx < rv; wcIdx++) {
    struct ibv_wc& wc = wcs[wcIdx];
    TP_THROW_ASSERT_IF(wc.status != IBV_WC_SUCCESS)
        << "Got failed work completion for queue pair " << wc.qp_num << ": "
        << wc.status;
    TP_THROW_ASSERT_IF(wc.qp_num != this->ibv.qp->qp_num);
    TP_THROW_ASSERT_IF(!(wc.wc_flags & IBV_WC_WITH_IMM));
    switch (wc.opcode) {
      case IBV_WC_RECV_RDMA_WITH_IMM:
        TP_THROW_ASSERT_IF(wc.imm_data != this->inbox.hostHead);
        handleRecv();
        isProgressing = true;
        break;
      case IBV_WC_RECV:
        TP_THROW_ASSERT_IF(wc.imm_data != this->outbox.hostTail);
        acknowledgeSend();
        isProgressing = true;
        break;
      case IBV_WC_RDMA_WRITE:
        break;
      case IBV_WC_SEND:
        break;
      default:
        TP_THROW_ASSERT() << "Unknown opcode: " << wc.opcode;
    }
  }

  // postRecvRequestsOnSRQ_(numRecvs); FIXME @nocommit

  return isProgressing;
}

bool Channel::Impl::progress() {
  bool isProgressing = false;

  // Try to advance the outbox's tail or the inbox's head. Since the IB queue
  // pair is the consumer of the outbox and the producer of the inbox, this
  // happens when the corresponding IB operations complete.
  if (pollIbv()) {
    isProgressing = true;
  }

  // Try to advance the outbox's head. The CUDA kernel is the producer of new
  // elements in this ringbuffer, so check whether it advanced the head.
  // An increase in the head means that the GPU is done working on that chunk of
  // the ringbuffer and is letting the host *acquire* control over it so it can
  // send it over InfiniBand.
  // The "acquire" memory order ensures that all reads/writes that will be
  // scheduled after reading the increased head cannot be reordered and
  // performed before the load, in which case they could read data that isn't
  // ready yet.
  if (this->outbox.hostHead < this->outbox.sharedHeadPtr->load(std::memory_order_acquire)) {
    performSend();
    isProgressing = true;
  }

  // Try to advance the inbox's tail. The CUDA kernel is the consumer of the
  // elements in this ringbuffer, so check whether it advanced the tail.
  // An increase in the head means that the GPU is done working on that chunk of
  // the ringbuffer and is letting the host *acquire* control over it so that a
  // new chunk can be received into it over InfiniBand.
  // The "acquire" memory order ensures that all reads/writes that will be
  // scheduled after reading the increased head cannot be reordered and
  // performed before the load, in which case they could read data that isn't
  // ready yet.
  if (this->inbox.hostTail < this->inbox.sharedTailPtr->load(std::memory_order_acquire)) {
    confirmRecv();
    isProgressing = true;
  }

  return isProgressing;
}

void Channel::setId(std::string id) {
  impl_->setId(std::move(id));
}

void Channel::Impl::setId(std::string id) {
  loop_.deferToLoop(
      [this, id{std::move(id)}]() mutable { setIdFromLoop_(std::move(id)); });
}

void Channel::Impl::setIdFromLoop_(std::string id) {
  TP_DCHECK(loop_.inLoop());
  TP_VLOG(4) << "Channel " << id_ << " was renamed to " << id;
  id_ = std::move(id);
}

void Channel::close() {
  impl_->close();
}

Channel::~Channel() {
  close();
}

void Channel::Impl::close() {
  loop_.deferToLoop([this]() { closeFromLoop_(); });
}

void Channel::Impl::closeFromLoop_() {
  TP_DCHECK(loop_.inLoop());
  TP_VLOG(4) << "Channel " << id_ << " is closing";
  setError_(TP_CREATE_ERROR(ChannelClosedError));
}

void Channel::Impl::setError_(Error error) {
  // Don't overwrite an error that's already set.
  if (error_ || !error) {
    return;
  }

  error_ = std::move(error);

  handleError_();
}

void Channel::Impl::handleError_() {
  TP_DCHECK(loop_.inLoop());
  TP_VLOG(5) << "Channel " << id_ << " is handling error " << error_.what();

  // TODO Find a way to also tell our CUDA kernels to terminate.

  // FIXME Is it safe to already fire all the callbacks?

  this->inbox.sharedHeadPtr->~atomic<uint64_t>();
  this->inbox.sharedTailPtr->~atomic<uint64_t>();

  this->inbox.deviceToHost.reset();
  this->inbox.hostToDevice.reset();

  this->inbox.stagingBufferMr.reset();
  this->inbox.stagingBuffer.reset();

  this->outbox.sharedHeadPtr->~atomic<uint64_t>();
  this->outbox.sharedTailPtr->~atomic<uint64_t>();

  this->outbox.hostToDevice.reset();
  this->outbox.deviceToHost.reset();

  this->outbox.stagingBufferMr.reset();
  this->outbox.stagingBuffer.reset();

  this->ibv.qp.reset();
  this->ibv.cq.reset();
  this->ibv.pd.reset();

  connection_->close();
}

} // namespace gdr
} // namespace channel
} // namespace tensorpipe
