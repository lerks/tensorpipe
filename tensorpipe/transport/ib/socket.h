/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <sys/socket.h>

#include <chrono>
#include <cstring>
#include <memory>

#include <tensorpipe/common/defs.h>
#include <tensorpipe/common/error.h>
#include <tensorpipe/common/error_macros.h>
#include <tensorpipe/common/optional.h>
#include <tensorpipe/transport/error.h>
#include <tensorpipe/transport/ib/fd.h>

namespace tensorpipe {
namespace transport {
namespace ib {

class Sockaddr final {
 public:
  static Sockaddr createAbstractUnixAddr(const std::string& name);

  inline const struct sockaddr* addr() const {
    return reinterpret_cast<const struct sockaddr*>(&addr_);
  }

  inline socklen_t addrlen() const {
    return addrlen_;
  }

  std::string str() const;

 private:
  explicit Sockaddr(struct sockaddr* addr, socklen_t addrlen);

  struct sockaddr_storage addr_;
  socklen_t addrlen_;
};

class Socket final : public Fd {
 public:
  [[nodiscard]] static std::tuple<Error, std::shared_ptr<Socket>>
  createForFamily(sa_family_t ai_family);

  explicit Socket(int fd) : Fd(fd) {}

  // Configure if the socket is blocking or not.
  [[nodiscard]] Error block(bool on);

  // Bind socket to address.
  [[nodiscard]] Error bind(const Sockaddr& addr);

  // Listen on socket.
  [[nodiscard]] Error listen(int backlog);

  // Accept new socket connecting to listening socket.
  [[nodiscard]] std::tuple<Error, std::shared_ptr<Socket>> accept();

  // Connect to address.
  [[nodiscard]] Error connect(const Sockaddr& addr);
};

} // namespace ib
} // namespace transport
} // namespace tensorpipe
