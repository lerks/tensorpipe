/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <tensorpipe/transport/ib/connection.h>

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cassert>
#include <cstring>
#include <deque>
#include <vector>

#include <google/protobuf/message_lite.h>

#include <tensorpipe/common/callback.h>
#include <tensorpipe/common/defs.h>
#include <tensorpipe/common/error_macros.h>
#include <tensorpipe/common/mem.h>
#include <tensorpipe/transport/error.h>
#include <tensorpipe/transport/ib/loop.h>
#include <tensorpipe/transport/ib/reactor.h>
#include <tensorpipe/transport/ib/socket.h>
#include <tensorpipe/util/ringbuffer/consumer.h>
#include <tensorpipe/util/ringbuffer/producer.h>
#include <tensorpipe/util/ringbuffer/protobuf_streams.h>
#include <tensorpipe/util/ringbuffer/shm.h>

namespace tensorpipe {
namespace transport {
namespace ib {

namespace {

IbvQueuePair createIbvQueuePair(
    IbvProtectionDomain& pd,
    IbvCompletionQueue& cq,
    IbvSharedReceiveQueue& srq) {
  struct ibv_qp_init_attr initAttr;
  std::memset(&initAttr, 0, sizeof(initAttr));
  initAttr.qp_type = IBV_QPT_RC;
  initAttr.send_cq = cq.ptr();
  initAttr.recv_cq = cq.ptr();
  initAttr.cap.max_send_wr = 1000;
  initAttr.cap.max_send_sge = 2;
  initAttr.srq = srq.ptr();
  initAttr.sq_sig_all = 1;
  return IbvQueuePair(pd, initAttr);
}

struct Exchange {
  IbvSetupInformation setupInfo;
  uint64_t memoryRegionPtr;
  uint32_t memoryRegionKey;
};

class ReadOperation {
  enum Mode {
    READ_LENGTH,
    READ_PAYLOAD,
  };

 public:
  using read_callback_fn = Connection::read_callback_fn;
  explicit ReadOperation(void* ptr, size_t len, read_callback_fn fn);
  explicit ReadOperation(read_callback_fn fn);

  // Processes a pending read.
  bool handleRead(util::ringbuffer::Consumer& consumer);

  bool completed() const {
    return (mode_ == READ_PAYLOAD && bytesRead_ == len_);
  }

  void handleError(const Error& error);

 private:
  Mode mode_{READ_LENGTH};
  void* ptr_{nullptr};
  std::unique_ptr<uint8_t[]> buf_;
  size_t len_{0};
  size_t bytesRead_{0};
  read_callback_fn fn_;
  const bool ptrProvided_;
};

class WriteOperation {
  enum Mode {
    WRITE_LENGTH,
    WRITE_PAYLOAD,
  };

 public:
  using write_callback_fn = Connection::write_callback_fn;
  WriteOperation(const void* ptr, size_t len, write_callback_fn fn);

  bool handleWrite(util::ringbuffer::Producer& producer);

  bool completed() const {
    return (mode_ == WRITE_PAYLOAD && bytesWritten_ == len_);
  }

  void handleError(const Error& error);

 private:
  Mode mode_{WRITE_LENGTH};
  const void* ptr_{nullptr};
  size_t len_{0};
  size_t bytesWritten_{0};
  write_callback_fn fn_;
};

ReadOperation::ReadOperation(void* ptr, size_t len, read_callback_fn fn)
    : ptr_(ptr), len_(len), fn_(std::move(fn)), ptrProvided_(true) {}

ReadOperation::ReadOperation(read_callback_fn fn)
    : fn_(std::move(fn)), ptrProvided_(false) {}

bool ReadOperation::handleRead(util::ringbuffer::Consumer& inbox) {
  // Start read transaction.
  // Retry because this must succeed.
  for (;;) {
    const auto ret = inbox.startTx();
    TP_DCHECK(ret >= 0 || ret == -EAGAIN);
    if (ret < 0) {
      continue;
    }
    break;
  }

  bool lengthRead = false;
  if (mode_ == READ_LENGTH) {
    uint32_t length;
    {
      ssize_t ret;
      ret = inbox.copyInTx(sizeof(length), &length);
      if (ret == -ENODATA) {
        ret = inbox.cancelTx();
        TP_THROW_SYSTEM_IF(ret < 0, -ret);
        return false;
      }
      TP_THROW_SYSTEM_IF(ret < 0, -ret);
    }

    if (ptrProvided_) {
      TP_DCHECK_EQ(length, len_);
    } else {
      len_ = length;
      buf_ = std::make_unique<uint8_t[]>(len_);
      ptr_ = buf_.get();
    }
    mode_ = READ_PAYLOAD;
    lengthRead = true;
  }

  // If reading empty buffer, skip payload read.
  if (len_ > 0) {
    const auto ret = inbox.copyAtMostInTx(
        len_ - bytesRead_, reinterpret_cast<uint8_t*>(ptr_) + bytesRead_);
    if (ret == -ENODATA) {
      if (lengthRead) {
        const auto ret = inbox.commitTx();
        TP_THROW_SYSTEM_IF(ret < 0, -ret);
        return true;
      } else {
        const auto ret = inbox.cancelTx();
        TP_THROW_SYSTEM_IF(ret < 0, -ret);
        return false;
      }
    }
    TP_THROW_SYSTEM_IF(ret < 0, -ret);
    bytesRead_ += ret;
  }

  {
    const auto ret = inbox.commitTx();
    TP_THROW_SYSTEM_IF(ret < 0, -ret);
  }

  if (completed()) {
    fn_(Error::kSuccess, ptr_, len_);
  }

  return true;
}

void ReadOperation::handleError(const Error& error) {
  fn_(error, nullptr, 0);
}

WriteOperation::WriteOperation(
    const void* ptr,
    size_t len,
    write_callback_fn fn)
    : ptr_(ptr), len_(len), fn_(std::move(fn)) {}

bool WriteOperation::handleWrite(util::ringbuffer::Producer& outbox) {
  // Start write transaction.
  // Retry because this must succeed.
  // TODO: fallback if it doesn't.
  // FIXME: In fact this must succeed *on the first try*, because this
  // connection is the only one writing to that outbox and it always does so
  // from the reactor thread, hence anyone else holding the lock is a big
  // error and we should make a big fuss.
  for (;;) {
    const auto ret = outbox.startTx();
    TP_DCHECK(ret >= 0 || ret == -EAGAIN);
    if (ret < 0) {
      continue;
    }
    break;
  }

  ssize_t ret;
  if (mode_ == WRITE_LENGTH) {
    ret = outbox.writeInTx<uint32_t>(len_);
    if (ret > 0) {
      mode_ = WRITE_PAYLOAD;
    }
  }

  // If writing empty buffer, skip payload write because ptr_
  // could be nullptr.
  if (mode_ == WRITE_PAYLOAD && len_ > 0) {
    ret = outbox.writeAtMostInTx(
        len_ - bytesWritten_,
        static_cast<const uint8_t*>(ptr_) + bytesWritten_);
    if (ret > 0) {
      bytesWritten_ += ret;
    }
  }

  if (ret == -ENOSPC) {
    const auto ret = outbox.cancelTx();
    TP_THROW_SYSTEM_IF(ret < 0, -ret);
    return false;
  }
  TP_THROW_SYSTEM_IF(ret < 0, -ret);

  {
    const auto ret = outbox.commitTx();
    TP_THROW_SYSTEM_IF(ret < 0, -ret);
  }

  if (completed()) {
    fn_(Error::kSuccess);
  }

  return true;
}

void WriteOperation::handleError(const Error& error) {
  fn_(error);
}

} // namespace

class Connection::Impl : public std::enable_shared_from_this<Connection::Impl>,
                         public EventHandler {
  static constexpr auto kBufferSize = 2 * 1024 * 1024;

  enum State {
    INITIALIZING = 1,
    SEND_ADDR,
    RECV_ADDR,
    ESTABLISHED,
  };

 public:
  // Create a connection that is already connected (e.g. from a listener).
  Impl(
      std::shared_ptr<Context::PrivateIface> context,
      std::shared_ptr<Socket> socket,
      std::string id);

  // Create a connection that connects to the specified address.
  Impl(
      std::shared_ptr<Context::PrivateIface> context,
      address_t addr,
      std::string id);

  // Initialize member fields that need `shared_from_this`.
  void init();

  // Queue a read operation.
  void read(read_callback_fn fn);
  void read(void* ptr, size_t length, read_callback_fn fn);

  // Perform a write operation.
  void write(const void* ptr, size_t length, write_callback_fn fn);

  // Tell the connection what its identifier is.
  void setId(std::string id);

  // Shut down the connection and its resources.
  void close();

  // Implementation of EventHandler.
  void handleEventsFromLoop(int events) override;

 private:
  // Initialize member fields that need `shared_from_this`.
  void initFromLoop();

  // Queue a read operation.
  void readFromLoop(read_callback_fn fn);
  void readFromLoop(
      google::protobuf::MessageLite& message,
      read_proto_callback_fn fn);
  void readFromLoop(void* ptr, size_t length, read_callback_fn fn);

  // Perform a write operation.
  void writeFromLoop(const void* ptr, size_t length, write_callback_fn fn);
  void writeFromLoop(
      const google::protobuf::MessageLite& message,
      write_callback_fn fn);

  void setIdFromLoop_(std::string id);

  // Shut down the connection and its resources.
  void closeFromLoop();

  // Handle events of type EPOLLIN on the UNIX domain socket.
  //
  // The only data that is expected on that socket is the file descriptors for
  // the other side's inbox (which is this side's outbox) and its reactor, plus
  // the reactor tokens to trigger the other side to read or write.
  void handleEventInFromLoop();

  // Handle events of type EPOLLOUT on the UNIX domain socket.
  //
  // Once the socket is writable we send the file descriptors for this side's
  // inbox (which the other side's outbox) and our reactor, plus the reactor
  // tokens to trigger this connection to read or write.
  void handleEventOutFromLoop();

  State state_{INITIALIZING};
  Error error_{Error::kSuccess};
  std::shared_ptr<Context::PrivateIface> context_;
  std::shared_ptr<Socket> socket_;
  optional<Sockaddr> sockaddr_;
  ClosingReceiver closingReceiver_;

  // Pending read operations.
  std::deque<ReadOperation> readOperations_;

  // Pending write operations.
  std::deque<WriteOperation> writeOperations_;

  // A sequence number for the calls to read.
  uint64_t nextBufferBeingRead_{0};

  // A sequence number for the calls to write.
  uint64_t nextBufferBeingWritten_{0};

  // A sequence number for the invocations of the callbacks of read.
  uint64_t nextReadCallbackToCall_{0};

  // A sequence number for the invocations of the callbacks of write.
  uint64_t nextWriteCallbackToCall_{0};

  // An identifier for the connection, composed of the identifier for the
  // context or listener, combined with an increasing sequence number. It will
  // only be used for logging and debugging purposes.
  std::string id_;

  // Process pending read operations if in an operational state.
  //
  // This may be triggered by the other side of the connection (by pushing this
  // side's inbox token to the reactor) when it has written some new data to its
  // outbox (which is this side's inbox). It is also called by this connection
  // when it moves into an established state or when a new read operation is
  // queued, in case data was already available before this connection was ready
  // to consume it.
  void processReadOperationsFromLoop();

  // Process pending write operations if in an operational state.
  //
  // This may be triggered by the other side of the connection (by pushing this
  // side's outbox token to the reactor) when it has read some data from its
  // inbox (which is this side's outbox). This is important when some of this
  // side's writes couldn't complete because the outbox was full, and thus they
  // needed to wait for some of its data to be read. This method is also called
  // by this connection when it moves into an established state, in case some
  // writes were queued before the connection was ready to process them, or when
  // a new write operation is queued.
  void processWriteOperationsFromLoop();

  void setError_(Error error);

  // Deal with an error.
  void handleError();

  IbvQueuePair qp_;
  IbvSetupInformation ibvSelfInfo_;

  // FIXME The header stores head and tail using atomics, but we don't need
  // that. Are we incurring in a perf penalty? Could we avoid it?
  util::ringbuffer::RingBufferHeader inboxHeader_{kBufferSize};
  util::ringbuffer::RingBufferHeader outboxHeader_{kBufferSize};
  PageAlignedPtr inboxBuf_{kBufferSize};
  PageAlignedPtr outboxBuf_{kBufferSize};
  util::ringbuffer::RingBuffer inboxRb_{&inboxHeader_, inboxBuf_.ptr()};
  util::ringbuffer::RingBuffer outboxRb_{&outboxHeader_, outboxBuf_.ptr()};
  IbvMemoryRegion inboxMr_{context_->getIbvPd(),
                           inboxBuf_.ptr(),
                           kBufferSize,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE};
  IbvMemoryRegion outboxMr_{context_->getIbvPd(),
                            outboxBuf_.ptr(),
                            kBufferSize,
                            0};

  uint32_t peerInboxKey_{0};
  uint64_t peerInboxPtr_{0};
};

Connection::Connection(
    ConstructorToken /* unused */,
    std::shared_ptr<Context::PrivateIface> context,
    std::shared_ptr<Socket> socket,
    std::string id)
    : impl_(std::make_shared<Impl>(
          std::move(context),
          std::move(socket),
          std::move(id))) {
  impl_->init();
}

Connection::Connection(
    ConstructorToken /* unused */,
    std::shared_ptr<Context::PrivateIface> context,
    address_t addr,
    std::string id)
    : impl_(std::make_shared<Impl>(
          std::move(context),
          std::move(addr),
          std::move(id))) {
  impl_->init();
}

void Connection::Impl::init() {
  context_->deferToLoop([impl{shared_from_this()}]() { impl->initFromLoop(); });
}

void Connection::close() {
  impl_->close();
}

void Connection::Impl::close() {
  context_->deferToLoop(
      [impl{shared_from_this()}]() { impl->closeFromLoop(); });
}

Connection::~Connection() {
  close();
}

Connection::Impl::Impl(
    std::shared_ptr<Context::PrivateIface> context,
    std::shared_ptr<Socket> socket,
    std::string id)
    : context_(std::move(context)),
      socket_(std::move(socket)),
      closingReceiver_(context_, context_->getClosingEmitter()),
      id_(std::move(id)) {}

Connection::Impl::Impl(
    std::shared_ptr<Context::PrivateIface> context,
    address_t addr,
    std::string id)
    : context_(std::move(context)),
      sockaddr_(Sockaddr::createAbstractUnixAddr(addr)),
      closingReceiver_(context_, context_->getClosingEmitter()),
      id_(std::move(id)) {}

void Connection::Impl::initFromLoop() {
  TP_DCHECK(context_->inLoopThread());

  closingReceiver_.activate(*this);

  Error error;
  // The connection either got a socket or an address, but not both.
  TP_DCHECK((socket_ != nullptr) ^ sockaddr_.has_value());
  if (socket_ == nullptr) {
    std::tie(error, socket_) = Socket::createForFamily(AF_UNIX);
    if (error) {
      setError_(std::move(error));
      return;
    }
    error = socket_->connect(sockaddr_.value());
    if (error) {
      setError_(std::move(error));
      return;
    }
  }
  // Ensure underlying control socket is non-blocking such that it
  // works well with event driven I/O.
  error = socket_->block(false);
  if (error) {
    setError_(std::move(error));
    return;
  }

  qp_ = createIbvQueuePair(
      context_->getIbvPd(), context_->getIbvCq(), context_->getIbvSrq());
  transitionIbvQueuePairToInit(qp_, context_->getIbvAddress());

  // We're sending file descriptors first, so wait for writability.
  state_ = SEND_ADDR;
  context_->registerDescriptor(socket_->fd(), EPOLLOUT, shared_from_this());

  context_->registerQp(
      qp_->qp_num,
      runIfAlive(
          *this,
          [](Impl& impl, uint32_t numRead) {
            TP_VLOG(9) << "Connection " << impl.id_ << " was signalled that "
                       << numRead << " bytes were read from its outbox";
            impl.outboxHeader_.incTail(numRead);
            impl.processWriteOperationsFromLoop();
          }),
      runIfAlive(*this, [](Impl& impl, uint32_t numWritten) {
        TP_VLOG(9) << "Connection " << impl.id_ << " was signalled that "
                   << numWritten << " bytes were written to its inbox";
        impl.inboxHeader_.incHead(numWritten);
        impl.processReadOperationsFromLoop();
      }));
}

void Connection::read(read_callback_fn fn) {
  impl_->read(std::move(fn));
}

void Connection::Impl::read(read_callback_fn fn) {
  context_->deferToLoop(
      [impl{shared_from_this()}, fn{std::move(fn)}]() mutable {
        impl->readFromLoop(std::move(fn));
      });
}

void Connection::Impl::readFromLoop(read_callback_fn fn) {
  TP_DCHECK(context_->inLoopThread());

  uint64_t sequenceNumber = nextBufferBeingRead_++;
  TP_VLOG(7) << "Connection " << id_ << " received an unsized read request (#"
             << sequenceNumber << ")";

  fn = [this, sequenceNumber, fn{std::move(fn)}](
           const Error& error, const void* ptr, size_t length) {
    TP_DCHECK_EQ(sequenceNumber, nextReadCallbackToCall_++);
    TP_VLOG(7) << "Connection " << id_
               << " is calling an unsized read callback (#" << sequenceNumber
               << ")";
    fn(error, ptr, length);
    TP_VLOG(7) << "Connection " << id_
               << " done calling an unsized read callback (#" << sequenceNumber
               << ")";
  };

  if (error_) {
    fn(error_, nullptr, 0);
    return;
  }

  readOperations_.emplace_back(std::move(fn));

  // If the inbox already contains some data, we may be able to process this
  // operation right away.
  processReadOperationsFromLoop();
}

void Connection::read(void* ptr, size_t length, read_callback_fn fn) {
  impl_->read(ptr, length, std::move(fn));
}

void Connection::Impl::read(void* ptr, size_t length, read_callback_fn fn) {
  context_->deferToLoop(
      [impl{shared_from_this()}, ptr, length, fn{std::move(fn)}]() mutable {
        impl->readFromLoop(ptr, length, std::move(fn));
      });
}

void Connection::Impl::readFromLoop(
    void* ptr,
    size_t length,
    read_callback_fn fn) {
  TP_DCHECK(context_->inLoopThread());

  uint64_t sequenceNumber = nextBufferBeingRead_++;
  TP_VLOG(7) << "Connection " << id_ << " received a sized read request (#"
             << sequenceNumber << ")";

  fn = [this, sequenceNumber, fn{std::move(fn)}](
           const Error& error, const void* ptr, size_t length) {
    TP_DCHECK_EQ(sequenceNumber, nextReadCallbackToCall_++);
    TP_VLOG(7) << "Connection " << id_ << " is calling a sized read callback (#"
               << sequenceNumber << ")";
    fn(error, ptr, length);
    TP_VLOG(7) << "Connection " << id_
               << " done calling a sized read callback (#" << sequenceNumber
               << ")";
  };

  if (error_) {
    fn(error_, ptr, length);
    return;
  }

  readOperations_.emplace_back(ptr, length, std::move(fn));

  // If the inbox already contains some data, we may be able to process this
  // operation right away.
  processReadOperationsFromLoop();
}

void Connection::write(const void* ptr, size_t length, write_callback_fn fn) {
  impl_->write(ptr, length, std::move(fn));
}

void Connection::Impl::write(
    const void* ptr,
    size_t length,
    write_callback_fn fn) {
  context_->deferToLoop(
      [impl{shared_from_this()}, ptr, length, fn{std::move(fn)}]() mutable {
        impl->writeFromLoop(ptr, length, std::move(fn));
      });
}

void Connection::Impl::writeFromLoop(
    const void* ptr,
    size_t length,
    write_callback_fn fn) {
  TP_DCHECK(context_->inLoopThread());

  uint64_t sequenceNumber = nextBufferBeingWritten_++;
  TP_VLOG(7) << "Connection " << id_ << " received a write request (#"
             << sequenceNumber << ")";

  fn = [this, sequenceNumber, fn{std::move(fn)}](const Error& error) {
    TP_DCHECK_EQ(sequenceNumber, nextWriteCallbackToCall_++);
    TP_VLOG(7) << "Connection " << id_ << " is calling a write callback (#"
               << sequenceNumber << ")";
    fn(error);
    TP_VLOG(7) << "Connection " << id_ << " done calling a write callback (#"
               << sequenceNumber << ")";
  };

  if (error_) {
    fn(error_);
    return;
  }

  writeOperations_.emplace_back(ptr, length, std::move(fn));

  // If the outbox has some free space, we may be able to process this operation
  // right away.
  processWriteOperationsFromLoop();
}

void Connection::setId(std::string id) {
  impl_->setId(std::move(id));
}

void Connection::Impl::setId(std::string id) {
  context_->deferToLoop(
      [impl{shared_from_this()}, id{std::move(id)}]() mutable {
        impl->setIdFromLoop_(std::move(id));
      });
}

void Connection::Impl::setIdFromLoop_(std::string id) {
  TP_DCHECK(context_->inLoopThread());
  TP_VLOG(7) << "Connection " << id_ << " was renamed to " << id;
  id_ = std::move(id);
}

void Connection::Impl::handleEventsFromLoop(int events) {
  TP_DCHECK(context_->inLoopThread());
  TP_VLOG(9) << "Connection " << id_ << " is handling an event on its socket ("
             << Loop::formatEpollEvents(events) << ")";

  // Handle only one of the events in the mask. Events on the control
  // file descriptor are rare enough for the cost of having epoll call
  // into this function multiple times to not matter. The benefit is
  // that every handler can close and unregister the control file
  // descriptor from the event loop, without worrying about the next
  // handler trying to do so as well.
  // In some cases the socket could be in a state where it's both in an error
  // state and readable/writable. If we checked for EPOLLIN or EPOLLOUT first
  // and then returned after handling them, we would keep doing so forever and
  // never reach the error handling. So we should keep the error check first.
  if (events & EPOLLERR) {
    int error;
    socklen_t errorlen = sizeof(error);
    int rv = getsockopt(
        socket_->fd(),
        SOL_SOCKET,
        SO_ERROR,
        reinterpret_cast<void*>(&error),
        &errorlen);
    if (rv == -1) {
      setError_(TP_CREATE_ERROR(SystemError, "getsockopt", rv));
    } else {
      setError_(TP_CREATE_ERROR(SystemError, "async error on socket", error));
    }
    return;
  }
  if (events & EPOLLIN) {
    handleEventInFromLoop();
    return;
  }
  if (events & EPOLLOUT) {
    handleEventOutFromLoop();
    return;
  }
  // Check for hangup last, as there could be cases where we get EPOLLHUP but
  // there's still data to be read from the socket, so we want to deal with that
  // before dealing with the hangup.
  if (events & EPOLLHUP) {
    setError_(TP_CREATE_ERROR(EOFError));
    return;
  }
}

void Connection::Impl::handleEventInFromLoop() {
  TP_DCHECK(context_->inLoopThread());
  if (state_ == RECV_ADDR) {
    struct Exchange ex;

    auto err = socket_->read(&ex, sizeof(ex));
    if (err != sizeof(ex)) {
      setError_(TP_CREATE_ERROR(ShortReadError, sizeof(ex), err));
      return;
    }

    transitionIbvQueuePairToReadyToReceive(
        qp_, context_->getIbvAddress(), ex.setupInfo);
    transitionIbvQueuePairToReadyToSend(qp_, ibvSelfInfo_);

    peerInboxKey_ = ex.memoryRegionKey;
    peerInboxPtr_ = ex.memoryRegionPtr;

    // The connection is usable now.
    state_ = ESTABLISHED;
    processWriteOperationsFromLoop();
    processReadOperationsFromLoop();
    return;
  }

  if (state_ == ESTABLISHED) {
    // We don't expect to read anything on this socket once the
    // connection has been established. If we do, assume it's a
    // zero-byte read indicating EOF.
    setError_(TP_CREATE_ERROR(EOFError));
    return;
  }

  TP_THROW_ASSERT() << "EPOLLIN event not handled in state " << state_;
}

void Connection::Impl::handleEventOutFromLoop() {
  TP_DCHECK(context_->inLoopThread());
  if (state_ == SEND_ADDR) {
    Exchange ex;
    ibvSelfInfo_ = getIbvSetupInformation(context_->getIbvAddress(), qp_);
    ex.setupInfo = ibvSelfInfo_;
    ex.memoryRegionPtr = reinterpret_cast<uint64_t>(inboxBuf_.ptr());
    ex.memoryRegionKey = inboxMr_->rkey;

    auto err = socket_->write(reinterpret_cast<void*>(&ex), sizeof(ex));
    if (err != sizeof(ex)) {
      setError_(TP_CREATE_ERROR(ShortWriteError, sizeof(ex), err));
      return;
    }

    // Sent our fds. Wait for fds from peer.
    state_ = RECV_ADDR;
    context_->registerDescriptor(socket_->fd(), EPOLLIN, shared_from_this());
    return;
  }

  TP_THROW_ASSERT() << "EPOLLOUT event not handled in state " << state_;
}

void Connection::Impl::processReadOperationsFromLoop() {
  TP_DCHECK(context_->inLoopThread());

  if (state_ != ESTABLISHED) {
    return;
  }

  util::ringbuffer::Consumer inboxConsumer(inboxRb_);
  while (!readOperations_.empty()) {
    // FIXME Wouldn't it be nice if handleWrite returned the number of bytes
    // it writes to the ringbuffer? It would save us from having to find that
    // out through the difference between the head before and after.
    uint64_t prevInboxTail = inboxHeader_.readTail();
    ReadOperation& op = readOperations_.front();
    if (op.handleRead(inboxConsumer)) {
      if (inboxHeader_.readTail() > prevInboxTail) {
        uint32_t len = inboxHeader_.readTail() - prevInboxTail;

        struct ibv_send_wr wr;
        memset(&wr, 0, sizeof(wr));
        wr.opcode = IBV_WR_SEND_WITH_IMM;
        wr.imm_data = len;

        struct ibv_send_wr* badWr = nullptr;
        TP_CHECK_IBV_INT(ibv_post_send(qp_.ptr(), &wr, &badWr));
        TP_THROW_ASSERT_IF(badWr != nullptr);
      }
    }
    if (op.completed()) {
      readOperations_.pop_front();
    } else {
      break;
    }
  }
}

void Connection::Impl::processWriteOperationsFromLoop() {
  TP_DCHECK(context_->inLoopThread());

  if (state_ != ESTABLISHED) {
    return;
  }

  util::ringbuffer::Producer outboxProducer(outboxRb_);
  while (!writeOperations_.empty()) {
    WriteOperation& op = writeOperations_.front();
    // FIXME Wouldn't it be nice if handleWrite returned the number of bytes
    // it writes to the ringbuffer? It would save us from having to find that
    // out through the difference between the head before and after.
    uint64_t prevOutboxHead = outboxHeader_.readHead();
    if (op.handleWrite(outboxProducer)) {
      if (outboxHeader_.readHead() > prevOutboxHead) {
        uint64_t start = prevOutboxHead % kBufferSize;
        uint64_t end = outboxHeader_.readHead() % kBufferSize;
        std::array<std::tuple<uint64_t, uint32_t>, 2> chunks;
        int numChunks;
        if (start < end) {
          numChunks = 1;
          chunks[0] = std::make_tuple(start, end - start);
        } else {
          // FIXME When end == 0 (== kBufferSize) this adds a useless chunk.
          numChunks = 2;
          chunks[0] = std::make_tuple(start, kBufferSize - start);
          chunks[1] = std::make_tuple(0, end);
        }

        for (int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
          uint64_t offset;
          uint32_t len;
          std::tie(offset, len) = chunks[chunkIdx];

          struct ibv_sge list;
          list.addr = reinterpret_cast<uint64_t>(outboxBuf_.ptr() + offset);
          list.length = len;
          list.lkey = outboxMr_->lkey;

          struct ibv_send_wr wr;
          std::memset(&wr, 0, sizeof(wr));
          wr.sg_list = &list;
          wr.num_sge = 1;
          wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
          wr.imm_data = len;
          wr.wr.rdma.remote_addr =
              static_cast<uint64_t>(peerInboxPtr_ + offset);
          wr.wr.rdma.rkey = peerInboxKey_;

          struct ibv_send_wr* badWr = nullptr;
          TP_CHECK_IBV_INT(ibv_post_send(qp_.ptr(), &wr, &badWr));
          TP_THROW_ASSERT_IF(badWr != nullptr);
        }
      }
    }
    if (op.completed()) {
      writeOperations_.pop_front();
    } else {
      break;
    }
  }
}

void Connection::Impl::setError_(Error error) {
  // Don't overwrite an error that's already set.
  if (error_ || !error) {
    return;
  }

  error_ = std::move(error);

  handleError();
}

void Connection::Impl::handleError() {
  TP_DCHECK(context_->inLoopThread());
  TP_VLOG(8) << "Connection " << id_ << " is handling error " << error_.what();

  for (auto& readOperation : readOperations_) {
    readOperation.handleError(error_);
  }
  readOperations_.clear();
  for (auto& writeOperation : writeOperations_) {
    writeOperation.handleError(error_);
  }
  writeOperations_.clear();
  context_->unregisterQp(qp_->qp_num);
  if (socket_ != nullptr) {
    if (state_ > INITIALIZING) {
      context_->unregisterDescriptor(socket_->fd());
    }
    socket_.reset();
  }
}

void Connection::Impl::closeFromLoop() {
  TP_DCHECK(context_->inLoopThread());
  TP_VLOG(7) << "Connection " << id_ << " is closing";
  setError_(TP_CREATE_ERROR(ConnectionClosedError));
}

} // namespace ib
} // namespace transport
} // namespace tensorpipe
