/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <tensorpipe/channel/gdr/context.h>

#include <algorithm>
#include <list>

#include <tensorpipe/channel/error.h>
#include <tensorpipe/channel/helpers.h>
#include <tensorpipe/channel/registry.h>
#include <tensorpipe/channel/gdr/channel.h>
#include <tensorpipe/channel/gdr/common.h>
#include <tensorpipe/common/callback.h>
#include <tensorpipe/common/defs.h>
#include <tensorpipe/common/error.h>
#include <tensorpipe/common/error_macros.h>

namespace tensorpipe {
namespace channel {
namespace gdr {

namespace {

std::shared_ptr<Context> makeGdrChannel() {
  return std::make_shared<Context>();
}

TP_REGISTER_CREATOR(TensorpipeChannelRegistry, gdr, makeGdrChannel);

} // namespace

class Context::Impl : public Context::PrivateIface,
                      public std::enable_shared_from_this<Context::Impl> {
 public:
  Impl();

  const std::string& domainDescriptor() const;

  std::shared_ptr<channel::Channel> createChannel(
      std::shared_ptr<transport::Connection>,
      Channel::Endpoint);

  void setId(std::string id);

  ClosingEmitter& getClosingEmitter() override;

  IbvContext& getIbvContext() override;

  IbvAddress& getIbvAddress() override;

  uint64_t registerProgressFn(std::function<bool()> fn);

  void deregisterProgressFn(uint64_t id);

  void close();

  void join();

  ~Impl() override = default;

 private:
  std::string domainDescriptor_;
  std::atomic<bool> closed_{false};
  std::atomic<bool> joined_{false};
  ClosingEmitter closingEmitter_;

  // An identifier for the context, composed of the identifier for the context,
  // combined with the channel's name. It will only be used for logging and
  // debugging purposes.
  std::string id_{"N/A"};

  // Sequence numbers for the channels created by this context, used to create
  // their identifiers based off this context's identifier. They will only be
  // used for logging and debugging.
  std::atomic<uint64_t> channelCounter_{0};

  IbvContext ibvContext_;
  IbvAddress ibvAddress_;

  void attend_();

  std::thread attendantThread_;

  std::unordered_map<uint64_t, std::function<bool()>> progressFns_;
  uint64_t nextProgressFnId_{0};
  std::mutex progressFnsMutex_;
};

Context::Context() : impl_(std::make_shared<Impl>()) {}

Context::Impl::Impl() : domainDescriptor_("any") {
  TP_CHECK_IBV_INT(ibv_fork_init());

  {
    IbvDeviceList ibvDeviceList;
    TP_THROW_ASSERT_IF(ibvDeviceList.size() == 0);

    for (int idx = 0; idx < ibvDeviceList.size(); ++idx) {
      std::string ibNicName(ibv_get_device_name(&ibvDeviceList[idx]));
      if (ibNicName == kIbvHca) {
        ibvContext_ = IbvContext(ibvDeviceList[idx]);
        break;
      }
    }
    TP_THROW_ASSERT_IF(ibvContext_.ptr() == nullptr)
        << "Couldn't find an Infiniband HCA called " << kIbvHca;

    ibvAddress_ = tensorpipe::getIbvAddress(
        ibvContext_, /*portNum=*/1, /*globalIdentifierIndex=*/0);
  }

  attendantThread_ = std::thread(&Context::Impl::attend_, this);
}

ClosingEmitter& Context::Impl::getClosingEmitter() {
  return closingEmitter_;
}

IbvContext& Context::Impl::getIbvContext() {
  return ibvContext_;
}

IbvAddress& Context::Impl::getIbvAddress() {
  return ibvAddress_;
}

uint64_t Context::Impl::registerProgressFn(std::function<bool()> fn) {
  std::unique_lock<std::mutex> lock(progressFnsMutex_);
  uint64_t id = nextProgressFnId_++;
  progressFns_.emplace(id, std::move(fn));
  return id;
}

void Context::Impl::deregisterProgressFn(uint64_t id) {
  std::unique_lock<std::mutex> lock(progressFnsMutex_);
  progressFns_.erase(id);
}

void Context::Impl::attend_() {
  while (!closed_) {
    std::unique_lock<std::mutex> lock(progressFnsMutex_);
    bool isProgressing = false;
    for (auto& iter : progressFns_) {
      if (iter.second()) {
        isProgressing = true;
      }
    }
    if (!isProgressing) {
      std::this_thread::yield();
    }
  }
}

const std::string& Context::domainDescriptor() const {
  return impl_->domainDescriptor();
}

const std::string& Context::Impl::domainDescriptor() const {
  return domainDescriptor_;
}

std::shared_ptr<channel::Channel> Context::createChannel(
    std::shared_ptr<transport::Connection> connection,
    Channel::Endpoint endpoint) {
  return impl_->createChannel(std::move(connection), endpoint);
}

std::shared_ptr<channel::Channel> Context::Impl::createChannel(
    std::shared_ptr<transport::Connection> connection,
    Channel::Endpoint /* unused */) {
  std::string channelId = id_ + ".c" + std::to_string(channelCounter_++);
  TP_VLOG(4) << "Channel context " << id_ << " is opening channel "
             << channelId;
  return std::make_shared<Channel>(
      Channel::ConstructorToken(),
      std::static_pointer_cast<PrivateIface>(shared_from_this()),
      std::move(connection),
      std::move(channelId));
}

void Context::close() {
  impl_->close();
}

void Context::Impl::close() {
  if (!closed_.exchange(true)) {
    TP_VLOG(4) << "Channel context " << id_ << " is closing";

    closingEmitter_.close();

    ibvContext_.reset();

    TP_VLOG(4) << "Channel context " << id_ << " done closing";
  }
}

void Context::join() {
  impl_->join();
}

void Context::Impl::join() {
  close();

  if (!joined_.exchange(true)) {
    TP_VLOG(4) << "Channel context " << id_ << " is joining";

    attendantThread_.join();

    TP_VLOG(4) << "Channel context " << id_ << " done joining";
  }
}

Context::~Context() {
  join();
}

void Context::setId(std::string id) {
  impl_->setId(std::move(id));
}

void Context::Impl::setId(std::string id) {
  TP_VLOG(4) << "Channel context " << id_ << " was renamed to " << id;
  id_ = std::move(id);
}

} // namespace gdr
} // namespace channel
} // namespace tensorpipe
