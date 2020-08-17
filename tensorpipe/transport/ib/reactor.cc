/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <tensorpipe/transport/ib/reactor.h>

#include <tensorpipe/common/ibv.h>
#include <tensorpipe/common/system.h>
#include <tensorpipe/util/ringbuffer/shm.h>

namespace tensorpipe {
namespace transport {
namespace ib {

Reactor::Reactor() {
  IbvDeviceList deviceList;
  TP_THROW_ASSERT_IF(deviceList.size() == 0);
  TP_LOG_ERROR() << deviceList[0].dev_path;
  TP_LOG_ERROR() << deviceList[0].ibdev_path;
  ctx_ = IbvContext(deviceList[0]);
  pd_ = IbvProtectionDomain(ctx_);
  cq_ = IbvCompletionQueue(ctx_, 1000, nullptr, nullptr, 0);

  struct ibv_srq_init_attr srqInitAttr;
  std::memset(&srqInitAttr, 0, sizeof(srqInitAttr));
  srqInitAttr.srq_context = nullptr;
  srqInitAttr.attr.max_wr = 1000;
  srqInitAttr.attr.max_sge = 0;
  srqInitAttr.attr.srq_limit = 0;
  srq_ = IbvSharedReceiveQueue(pd_, srqInitAttr);

  addr_ = tensorpipe::getIbvAddress(
      ctx_, /*portNum=*/1, /*globalIdenfifierIndex=*/0);

  postRecvRequestsOnSRQ_(1000);

  thread_ = std::thread(&Reactor::run, this);
}

void Reactor::postRecvRequestsOnSRQ_(int num) {
  if (num == 0) {
    return;
  }
  struct ibv_recv_wr* badRecvWr = nullptr;
  // FIXME Use a smaller array size and do multiple passes if needed.
  std::array<struct ibv_recv_wr, 1000> wrs;
  std::memset(wrs.data(), 0, sizeof(wrs));
  for (int i = 0; i < num - 1; i++) {
    wrs[i].next = &wrs[i + 1];
  }
  int rv = ibv_post_srq_recv(srq_.ptr(), wrs.data(), &badRecvWr);
  TP_THROW_SYSTEM_IF(rv != 0, errno);
  TP_THROW_ASSERT_IF(badRecvWr != nullptr);
}

void Reactor::close() {
  if (!closed_.exchange(true)) {
    // No need to wake up the reactor, since it is busy-waiting.
  }
}

void Reactor::join() {
  close();

  if (!joined_.exchange(true)) {
    thread_.join();
  }
}

Reactor::~Reactor() {
  join();
}

void Reactor::run() {
  setThreadName("TP_SHM_reactor");

  // Stop when another thread has asked the reactor the close and when
  // all functions have been removed.
  while (!closed_ || fns_.size() > 0) {
    std::array<struct ibv_wc, 1> wcs;
    auto rv = ibv_poll_cq(cq_.ptr(), wcs.size(), wcs.data());
    TP_THROW_SYSTEM_IF(rv < 0, errno);

    if (rv == 0) {
      if (deferredFunctionCount_ > 0) {
        decltype(deferredFunctionList_) fns;

        {
          std::unique_lock<std::mutex> lock(deferredFunctionMutex_);
          std::swap(fns, deferredFunctionList_);
        }

        deferredFunctionCount_ -= fns.size();

        for (auto& fn : fns) {
          fn();
        }
      } else {
        std::this_thread::yield();
      }
      continue;
    }

    int numRecvs = 0;
    for (int wcIdx = 0; wcIdx < rv; wcIdx++) {
      struct ibv_wc& wc = wcs[wcIdx];
      TP_THROW_ASSERT_IF(wc.status != IBV_WC_SUCCESS)
          << "Got failed work completion for queue pair " << wc.qp_num << ": "
          << wc.status;
      auto iter = fns_.find(wc.qp_num);
      if (iter == fns_.end()) {
        TP_VLOG(9) << "Got work completion for unknown queue pair "
                   << wc.qp_num;
        continue;
      }
      TP_THROW_ASSERT_IF(!(wc.wc_flags & IBV_WC_WITH_IMM));
      switch (wc.opcode) {
        case IBV_WC_RECV_RDMA_WITH_IMM: {
          auto& onProducedFn = std::get<1>(iter->second);
          onProducedFn(wc.imm_data);
        }
          numRecvs++;
          break;
        case IBV_WC_RECV: {
          auto& onConsumedFn = std::get<0>(iter->second);
          onConsumedFn(wc.imm_data);
        }
          numRecvs++;
          break;
        case IBV_WC_SEND:
          break;
        case IBV_WC_RDMA_WRITE:
          break;
        default:
          TP_THROW_ASSERT() << "Unknown opcode: " << wc.opcode;
      }
    }

    postRecvRequestsOnSRQ_(numRecvs);
  }

  // The loop is winding down and "handing over" control to the on demand loop.
  // But it can only do so safely once there are no pending deferred functions,
  // as otherwise those may risk never being executed.
  while (true) {
    decltype(deferredFunctionList_) fns;

    {
      std::unique_lock<std::mutex> lock(deferredFunctionMutex_);
      if (deferredFunctionList_.empty()) {
        isThreadConsumingDeferredFunctions_ = false;
        break;
      }
      std::swap(fns, deferredFunctionList_);
    }

    for (auto& fn : fns) {
      fn();
    }
  }
}

void Reactor::registerQp(
    uint32_t qpn,
    std::function<void(uint32_t)> onConsumed,
    std::function<void(uint32_t)> onProduced) {
  fns_.emplace(
      qpn, std::make_pair(std::move(onConsumed), std::move(onProduced)));
}

void Reactor::unregisterQp(uint32_t qpn) {
  fns_.erase(qpn);
}

void Reactor::deferToLoop(TDeferredFunction fn) {
  {
    std::unique_lock<std::mutex> lock(deferredFunctionMutex_);
    if (likely(isThreadConsumingDeferredFunctions_)) {
      deferredFunctionList_.push_back(std::move(fn));
      ++deferredFunctionCount_;
      // No need to wake up the reactor, since it is busy-waiting.
      return;
    }
  }
  // Must call it without holding the lock, as it could cause a reentrant call.
  onDemandLoop_.deferToLoop(std::move(fn));
}

} // namespace ib
} // namespace transport
} // namespace tensorpipe
