/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <tensorpipe/test/channel/channel_test.h>

#include <numeric>

using namespace tensorpipe;
using namespace tensorpipe::channel;

// Implement this in a subprocess as in some cases it may initialize CUDA and
// thus would otherwise "pollute" the parent process.
class DeviceDescriptorsTest : public ChannelTestCase {
 public:
  void run(ChannelTestHelper* helper) override {
    auto peerGroup = helper->makePeerGroup();
    peerGroup->spawn(
        [&] {
          std::shared_ptr<Context> context1 = helper->makeContext("ctx1");
          std::shared_ptr<Context> context2 = helper->makeContext("ctx2");
          const auto& descriptors1 = context1->deviceDescriptors();
          const auto& descriptors2 = context2->deviceDescriptors();

          EXPECT_FALSE(descriptors1.empty());
          EXPECT_FALSE(descriptors2.empty());

          EXPECT_EQ(descriptors1.size(), descriptors2.size());
          for (const auto& deviceIter : descriptors1) {
            EXPECT_FALSE(deviceIter.second.empty());
            EXPECT_EQ(descriptors2.count(deviceIter.first), 1);
            EXPECT_EQ(deviceIter.second, descriptors2.at(deviceIter.first));
          }
        },
        [] {});
  }
};

CHANNEL_TEST(ChannelTestSuite, DeviceDescriptors);

class ClientToServerTest : public ClientServerChannelTestCase {
 public:
  static constexpr int kDataSize = 256;

  void server(std::shared_ptr<Channel> channel) override {
    // Initialize with sequential values.
    std::vector<uint8_t> data(kDataSize);
    std::iota(data.begin(), data.end(), 0);
    std::unique_ptr<DataWrapper> wrappedData = helper_->makeDataWrapper(data);

    // Perform send and wait for completion.
    std::future<std::tuple<Error, TDescriptor>> descriptorFuture;
    std::future<Error> sendFuture;
    std::tie(descriptorFuture, sendFuture) =
        sendWithFuture(channel, wrappedData->buffer());
    Error descriptorError;
    TDescriptor descriptor;
    std::tie(descriptorError, descriptor) = descriptorFuture.get();
    EXPECT_FALSE(descriptorError) << descriptorError.what();
    this->peers_->send(PeerGroup::kClient, descriptor);
    Error sendError = sendFuture.get();
    EXPECT_FALSE(sendError) << sendError.what();

    this->peers_->done(PeerGroup::kServer);
    this->peers_->join(PeerGroup::kServer);
  }

  void client(std::shared_ptr<Channel> channel) override {
    std::unique_ptr<DataWrapper> wrappedData =
        helper_->makeDataWrapper(kDataSize);

    // Perform recv and wait for completion.
    auto descriptor = this->peers_->recv(PeerGroup::kClient);
    std::future<Error> recvFuture =
        recvWithFuture(channel, descriptor, wrappedData->buffer());
    Error recvError = recvFuture.get();
    EXPECT_FALSE(recvError) << recvError.what();

    // Validate contents of vector.
    auto unwrappedData = wrappedData->unwrap();
    for (auto i = 0; i < kDataSize; i++) {
      EXPECT_EQ(unwrappedData[i], i);
    }

    this->peers_->done(PeerGroup::kClient);
    this->peers_->join(PeerGroup::kClient);
  }
};

CHANNEL_TEST(ChannelTestSuite, ClientToServer);

class ServerToClientTest : public ClientServerChannelTestCase {
  static constexpr int kDataSize = 256;

 public:
  void server(std::shared_ptr<Channel> channel) override {
    std::unique_ptr<DataWrapper> wrappedData =
        helper_->makeDataWrapper(kDataSize);

    // Perform recv and wait for completion.
    auto descriptor = this->peers_->recv(PeerGroup::kServer);
    std::future<Error> recvFuture =
        recvWithFuture(channel, descriptor, wrappedData->buffer());
    Error recvError = recvFuture.get();
    EXPECT_FALSE(recvError) << recvError.what();

    // Validate contents of vector.
    auto unwrappedData = wrappedData->unwrap();
    for (auto i = 0; i < kDataSize; i++) {
      EXPECT_EQ(unwrappedData[i], i);
    }

    this->peers_->done(PeerGroup::kServer);
    this->peers_->join(PeerGroup::kServer);
  }

  void client(std::shared_ptr<Channel> channel) override {
    // Initialize with sequential values.
    std::vector<uint8_t> data(kDataSize);
    std::iota(data.begin(), data.end(), 0);
    std::unique_ptr<DataWrapper> wrappedData = helper_->makeDataWrapper(data);

    // Perform send and wait for completion.
    std::future<std::tuple<Error, TDescriptor>> descriptorFuture;
    std::future<Error> sendFuture;
    std::tie(descriptorFuture, sendFuture) =
        sendWithFuture(channel, wrappedData->buffer());
    Error descriptorError;
    TDescriptor descriptor;
    std::tie(descriptorError, descriptor) = descriptorFuture.get();
    EXPECT_FALSE(descriptorError) << descriptorError.what();
    this->peers_->send(PeerGroup::kServer, descriptor);
    Error sendError = sendFuture.get();
    EXPECT_FALSE(sendError) << sendError.what();

    this->peers_->done(PeerGroup::kClient);
    this->peers_->join(PeerGroup::kClient);
  }
};

CHANNEL_TEST(ChannelTestSuite, ServerToClient);

class SendMultipleTensorsTest : public ClientServerChannelTestCase {
  // FIXME This is very puzzling, as in CircleCI making this field static (and
  // possibly even constexpr) causes a undefined symbol link error.
  const int dataSize_ = 256 * 1024; // 256KB
  static constexpr int kNumTensors = 100;

 public:
  void server(std::shared_ptr<Channel> channel) override {
    // Initialize with sequential values.
    std::vector<uint8_t> data(dataSize_);
    std::iota(data.begin(), data.end(), 0);
    std::unique_ptr<DataWrapper> wrappedData = helper_->makeDataWrapper(data);

    // Error futures
    std::vector<std::future<Error>> sendFutures;

    // Perform send and wait for completion.
    for (int i = 0; i < kNumTensors; i++) {
      std::future<std::tuple<Error, TDescriptor>> descriptorFuture;
      std::future<Error> sendFuture;
      std::tie(descriptorFuture, sendFuture) =
          sendWithFuture(channel, wrappedData->buffer());
      Error descriptorError;
      TDescriptor descriptor;
      std::tie(descriptorError, descriptor) = descriptorFuture.get();
      EXPECT_FALSE(descriptorError) << descriptorError.what();
      this->peers_->send(PeerGroup::kClient, descriptor);
      sendFutures.push_back(std::move(sendFuture));
    }
    for (auto& sendFuture : sendFutures) {
      Error sendError = sendFuture.get();
      EXPECT_FALSE(sendError) << sendError.what();
    }

    this->peers_->done(PeerGroup::kServer);
    this->peers_->join(PeerGroup::kServer);
  }

  void client(std::shared_ptr<Channel> channel) override {
    std::vector<std::unique_ptr<DataWrapper>> wrappedDataVec;
    for (int i = 0; i < kNumTensors; i++) {
      wrappedDataVec.push_back(helper_->makeDataWrapper(dataSize_));
    }

    // Error futures
    std::vector<std::future<Error>> recvFutures;

    // Perform recv and wait for completion.
    for (auto& wrappedData : wrappedDataVec) {
      auto descriptor = this->peers_->recv(PeerGroup::kClient);
      std::future<Error> recvFuture =
          recvWithFuture(channel, descriptor, wrappedData->buffer());
      recvFutures.push_back(std::move(recvFuture));
    }
    for (auto& recvFuture : recvFutures) {
      Error recvError = recvFuture.get();
      EXPECT_FALSE(recvError) << recvError.what();
    }

    // Validate contents of vector.
    for (auto& wrappedData : wrappedDataVec) {
      auto unwrappedData = wrappedData->unwrap();
      for (int i = 0; i < dataSize_; i++) {
        EXPECT_EQ(unwrappedData[i], i % 256);
      }
    }

    this->peers_->done(PeerGroup::kClient);
    this->peers_->join(PeerGroup::kClient);
  }
};

CHANNEL_TEST(ChannelTestSuite, SendMultipleTensors);

class SendTensorsBothWaysTest : public ClientServerChannelTestCase {
  static constexpr int kDataSize = 256;

  void server(std::shared_ptr<Channel> channel) override {
    // Initialize sendBuffer with sequential values.
    std::vector<uint8_t> sendData(kDataSize);
    std::iota(sendData.begin(), sendData.end(), 0);
    std::unique_ptr<DataWrapper> wrappedSendData =
        helper_->makeDataWrapper(sendData);

    // Recv buffer.
    std::unique_ptr<DataWrapper> wrappedRecvData =
        helper_->makeDataWrapper(kDataSize);

    std::future<Error> sendFuture;
    std::future<Error> recvFuture;

    // Perform send.
    {
      std::future<std::tuple<Error, TDescriptor>> descriptorFuture;
      std::tie(descriptorFuture, sendFuture) =
          sendWithFuture(channel, wrappedSendData->buffer());
      Error descriptorError;
      TDescriptor descriptor;
      std::tie(descriptorError, descriptor) = descriptorFuture.get();
      EXPECT_FALSE(descriptorError) << descriptorError.what();
      this->peers_->send(PeerGroup::kClient, descriptor);
    }

    // Perform recv.
    {
      auto descriptor = this->peers_->recv(PeerGroup::kServer);
      recvFuture =
          recvWithFuture(channel, descriptor, wrappedRecvData->buffer());
    }

    // Wait for completion of both.
    Error sendError = sendFuture.get();
    EXPECT_FALSE(sendError) << sendError.what();
    Error recvError = recvFuture.get();
    EXPECT_FALSE(recvError) << recvError.what();

    // Verify recvd buffers.
    auto unwrappedData = wrappedRecvData->unwrap();
    for (int i = 0; i < kDataSize; i++) {
      EXPECT_EQ(unwrappedData[i], i % 256);
    }

    this->peers_->done(PeerGroup::kServer);
    this->peers_->join(PeerGroup::kServer);
  }

  void client(std::shared_ptr<Channel> channel) override {
    // Initialize sendBuffer with sequential values.
    std::vector<uint8_t> sendData(kDataSize);
    std::iota(sendData.begin(), sendData.end(), 0);
    std::unique_ptr<DataWrapper> wrappedSendData =
        helper_->makeDataWrapper(sendData);

    // Recv buffer.
    std::unique_ptr<DataWrapper> wrappedRecvData =
        helper_->makeDataWrapper(kDataSize);

    std::future<Error> sendFuture;
    std::future<Error> recvFuture;

    // Perform send.
    {
      std::future<std::tuple<Error, TDescriptor>> descriptorFuture;
      std::tie(descriptorFuture, sendFuture) =
          sendWithFuture(channel, wrappedSendData->buffer());
      Error descriptorError;
      TDescriptor descriptor;
      std::tie(descriptorError, descriptor) = descriptorFuture.get();
      EXPECT_FALSE(descriptorError) << descriptorError.what();
      this->peers_->send(PeerGroup::kServer, descriptor);
    }

    // Perform recv.
    {
      auto descriptor = this->peers_->recv(PeerGroup::kClient);
      recvFuture =
          recvWithFuture(channel, descriptor, wrappedRecvData->buffer());
    }

    // Wait for completion of both.
    Error sendError = sendFuture.get();
    EXPECT_FALSE(sendError) << sendError.what();
    Error recvError = recvFuture.get();
    EXPECT_FALSE(recvError) << recvError.what();

    // Verify recvd buffers.
    auto unwrappedData = wrappedRecvData->unwrap();
    for (int i = 0; i < kDataSize; i++) {
      EXPECT_EQ(unwrappedData[i], i % 256);
    }

    this->peers_->done(PeerGroup::kClient);
    this->peers_->join(PeerGroup::kClient);
  }
};

CHANNEL_TEST(ChannelTestSuite, SendTensorsBothWays);
