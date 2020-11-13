/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <utility>

#include <tensorpipe/common/callback.h>
#include <tensorpipe/common/defs.h>
#include <tensorpipe/common/error.h>
#include <tensorpipe/common/error_macros.h>
#include <tensorpipe/transport/connection.h>
#include <tensorpipe/transport/error.h>

namespace tensorpipe {
namespace transport {

template <typename TCtx, typename TList, typename TConn>
class ContextImplBoilerplate;

template <typename TCtx, typename TList, typename TConn>
class ListenerImplBoilerplate;

template <typename TCtx, typename TList, typename TConn>
class ConnectionImplBoilerplate : public std::enable_shared_from_this<TConn> {
 public:
  class ConstructorToken {
   public:
    ConstructorToken(const ConstructorToken&) = default;

   private:
    explicit ConstructorToken() {}
    friend ContextImplBoilerplate<TCtx, TList, TConn>;
    friend ListenerImplBoilerplate<TCtx, TList, TConn>;
  };

  ConnectionImplBoilerplate(
      ConstructorToken,
      std::shared_ptr<TCtx> context,
      std::string id);

  ConnectionImplBoilerplate(const ConnectionImplBoilerplate&) = delete;
  ConnectionImplBoilerplate(ConnectionImplBoilerplate&&) = delete;
  ConnectionImplBoilerplate& operator=(const ConnectionImplBoilerplate&) =
      delete;
  ConnectionImplBoilerplate& operator=(ConnectionImplBoilerplate&&) = delete;

  // Initialize member fields that need `shared_from_this`.
  void init();

  // Queue a read operation.
  using read_callback_fn = Connection::read_callback_fn;
  void read(read_callback_fn fn);
  void read(void* ptr, size_t length, read_callback_fn fn);

  // Perform a write operation.
  using write_callback_fn = Connection::write_callback_fn;
  void write(const void* ptr, size_t length, write_callback_fn fn);

  // Tell the connection what its identifier is.
  void setId(std::string id);

  // Shut down the connection and its resources.
  void close();

  virtual ~ConnectionImplBoilerplate() = default;

 protected:
  virtual void initImplFromLoop() = 0;
  virtual void readImplFromLoop(read_callback_fn fn) = 0;
  virtual void readImplFromLoop(
      void* ptr,
      size_t length,
      read_callback_fn fn) = 0;
  virtual void writeImplFromLoop(
      const void* ptr,
      size_t length,
      write_callback_fn fn) = 0;
  virtual void handleErrorImpl() = 0;

  void setError(Error error);

  const std::shared_ptr<TCtx> context_;

  Error error_{Error::kSuccess};

  // An identifier for the connection, composed of the identifier for the
  // context or listener, combined with an increasing sequence number. It will
  // only be used for logging and debugging purposes.
  std::string id_;

 private:
  // Initialize member fields that need `shared_from_this`.
  void initFromLoop();

  // Queue a read operation.
  void readFromLoop(read_callback_fn fn);
  void readFromLoop(void* ptr, size_t length, read_callback_fn fn);

  // Perform a write operation.
  void writeFromLoop(const void* ptr, size_t length, write_callback_fn fn);

  void setIdFromLoop(std::string id);

  // Shut down the connection and its resources.
  void closeFromLoop();

  // Deal with an error.
  void handleError();

  ClosingReceiver closingReceiver_;

  // A sequence number for the calls to read and write.
  uint64_t nextBufferBeingRead_{0};
  uint64_t nextBufferBeingWritten_{0};

  // A sequence number for the invocations of the callbacks of read and write.
  uint64_t nextReadCallbackToCall_{0};
  uint64_t nextWriteCallbackToCall_{0};
};

template <typename TCtx, typename TList, typename TConn>
ConnectionImplBoilerplate<TCtx, TList, TConn>::ConnectionImplBoilerplate(
    ConstructorToken /* unused */,
    std::shared_ptr<TCtx> context,
    std::string id)
    : context_(std::move(context)),
      id_(std::move(id)),
      closingReceiver_(context_, context_->getClosingEmitter()) {}

template <typename TCtx, typename TList, typename TConn>
void ConnectionImplBoilerplate<TCtx, TList, TConn>::init() {
  context_->deferToLoop(
      [impl{this->shared_from_this()}]() { impl->initFromLoop(); });
}

template <typename TCtx, typename TList, typename TConn>
void ConnectionImplBoilerplate<TCtx, TList, TConn>::initFromLoop() {
  closingReceiver_.activate(*this);

  initImplFromLoop();
}

template <typename TCtx, typename TList, typename TConn>
void ConnectionImplBoilerplate<TCtx, TList, TConn>::read(read_callback_fn fn) {
  context_->deferToLoop(
      [impl{this->shared_from_this()}, fn{std::move(fn)}]() mutable {
        impl->readFromLoop(std::move(fn));
      });
}

template <typename TCtx, typename TList, typename TConn>
void ConnectionImplBoilerplate<TCtx, TList, TConn>::readFromLoop(
    read_callback_fn fn) {
  TP_DCHECK(context_->inLoop());

  uint64_t sequenceNumber = nextBufferBeingRead_++;
  TP_VLOG(7) << "Connection " << id_ << " received a read request (#"
             << sequenceNumber << ")";

  fn = [this, sequenceNumber, fn{std::move(fn)}](
           const Error& error, const void* ptr, size_t length) {
    TP_DCHECK_EQ(sequenceNumber, nextReadCallbackToCall_++);
    TP_VLOG(7) << "Connection " << id_ << " is calling a read callback (#"
               << sequenceNumber << ")";
    fn(error, ptr, length);
    TP_VLOG(7) << "Connection " << id_ << " done calling a read callback (#"
               << sequenceNumber << ")";
  };

  if (error_) {
    fn(error_, nullptr, 0);
    return;
  }

  readImplFromLoop(std::move(fn));
}

template <typename TCtx, typename TList, typename TConn>
void ConnectionImplBoilerplate<TCtx, TList, TConn>::read(
    void* ptr,
    size_t length,
    read_callback_fn fn) {
  context_->deferToLoop([impl{this->shared_from_this()},
                         ptr,
                         length,
                         fn{std::move(fn)}]() mutable {
    impl->readFromLoop(ptr, length, std::move(fn));
  });
}

template <typename TCtx, typename TList, typename TConn>
void ConnectionImplBoilerplate<TCtx, TList, TConn>::readFromLoop(
    void* ptr,
    size_t length,
    read_callback_fn fn) {
  TP_DCHECK(context_->inLoop());

  uint64_t sequenceNumber = nextBufferBeingRead_++;
  TP_VLOG(7) << "Connection " << id_ << " received a read request (#"
             << sequenceNumber << ")";

  fn = [this, sequenceNumber, fn{std::move(fn)}](
           const Error& error, const void* ptr, size_t length) {
    TP_DCHECK_EQ(sequenceNumber, nextReadCallbackToCall_++);
    TP_VLOG(7) << "Connection " << id_ << " is calling a read callback (#"
               << sequenceNumber << ")";
    fn(error, ptr, length);
    TP_VLOG(7) << "Connection " << id_ << " done calling a read callback (#"
               << sequenceNumber << ")";
  };

  if (error_) {
    fn(error_, ptr, length);
    return;
  }

  readImplFromLoop(ptr, length, std::move(fn));
}

template <typename TCtx, typename TList, typename TConn>
void ConnectionImplBoilerplate<TCtx, TList, TConn>::write(
    const void* ptr,
    size_t length,
    write_callback_fn fn) {
  context_->deferToLoop([impl{this->shared_from_this()},
                         ptr,
                         length,
                         fn{std::move(fn)}]() mutable {
    impl->writeFromLoop(ptr, length, std::move(fn));
  });
}

template <typename TCtx, typename TList, typename TConn>
void ConnectionImplBoilerplate<TCtx, TList, TConn>::writeFromLoop(
    const void* ptr,
    size_t length,
    write_callback_fn fn) {
  TP_DCHECK(context_->inLoop());

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

  writeImplFromLoop(ptr, length, std::move(fn));
}

template <typename TCtx, typename TList, typename TConn>
void ConnectionImplBoilerplate<TCtx, TList, TConn>::setId(std::string id) {
  context_->deferToLoop(
      [impl{this->shared_from_this()}, id{std::move(id)}]() mutable {
        impl->setIdFromLoop(std::move(id));
      });
}

template <typename TCtx, typename TList, typename TConn>
void ConnectionImplBoilerplate<TCtx, TList, TConn>::setIdFromLoop(
    std::string id) {
  TP_DCHECK(context_->inLoop());
  TP_VLOG(7) << "Connection " << id_ << " was renamed to " << id;
  id_ = std::move(id);
}

template <typename TCtx, typename TList, typename TConn>
void ConnectionImplBoilerplate<TCtx, TList, TConn>::close() {
  context_->deferToLoop(
      [impl{this->shared_from_this()}]() { impl->closeFromLoop(); });
}

template <typename TCtx, typename TList, typename TConn>
void ConnectionImplBoilerplate<TCtx, TList, TConn>::closeFromLoop() {
  TP_DCHECK(context_->inLoop());
  TP_VLOG(7) << "Connection " << id_ << " is closing";
  setError(TP_CREATE_ERROR(ConnectionClosedError));
}

template <typename TCtx, typename TList, typename TConn>
void ConnectionImplBoilerplate<TCtx, TList, TConn>::setError(Error error) {
  // Don't overwrite an error that's already set.
  if (error_ || !error) {
    return;
  }

  error_ = std::move(error);

  handleError();
}

template <typename TCtx, typename TList, typename TConn>
void ConnectionImplBoilerplate<TCtx, TList, TConn>::handleError() {
  TP_DCHECK(context_->inLoop());
  TP_VLOG(8) << "Connection " << id_ << " is handling error " << error_.what();

  handleErrorImpl();
}

} // namespace transport
} // namespace tensorpipe
