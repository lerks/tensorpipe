/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>

#include <tensorpipe/proto/core.pb.h>
#include <tensorpipe/util/ringbuffer/consumer.h>
#include <tensorpipe/util/ringbuffer/producer.h>
#include <tensorpipe/util/ringbuffer/protobuf_streams.h>
#include <tensorpipe/util/ringbuffer/ringbuffer.h>

#include <google/protobuf/util/message_differencer.h>
#include <gtest/gtest.h>

using namespace tensorpipe;
using namespace tensorpipe::util::ringbuffer;

namespace {

// Holds and owns the memory for the ringbuffer's header and data.
class RingBufferStorage {
 public:
  RingBufferStorage(size_t size) : header_(size) {}

  RingBuffer getRb() {
    return {&header_, data_.get()};
  }

 private:
  RingBufferHeader header_;
  std::unique_ptr<uint8_t[]> data_ =
      std::make_unique<uint8_t[]>(header_.kDataPoolByteSize);
};

proto::Brochure getMessage() {
  proto::Brochure message;
  const std::map<std::string, std::string> ta_map = {{"foo", "bar"},
                                                     {"foobar", "baz"}};

  for (const auto& e : ta_map) {
    proto::TransportAdvertisement ta;
    ta.set_domain_descriptor(e.second);
    (*message.mutable_transport_advertisement())[e.first] = ta;
  }

  const std::map<std::string, std::string> ca_map = {{"foo2", "bar2"},
                                                     {"foobar2", "baz2"}};
  for (const auto& e : ca_map) {
    proto::ChannelAdvertisement ca;
    ca.set_domain_descriptor(e.second);
    (*message.mutable_channel_advertisement())[e.first] = ca;
  }

  return message;
}

void test_serialize_parse(RingBuffer rb, const proto::Brochure message) {
  uint32_t len = message.ByteSize();
  EXPECT_LE(len, rb.getHeader().kDataPoolByteSize)
      << "Message larger than ring buffer.";
  {
    Producer p{rb};

    ssize_t ret = p.startTx();
    EXPECT_EQ(ret, 0);

    ZeroCopyOutputStream os(&p, len);
    ASSERT_TRUE(message.SerializeToZeroCopyStream(&os));

    ret = p.commitTx();
    EXPECT_EQ(ret, 0);

    EXPECT_EQ(rb.getHeader().usedSizeWeak(), len);
  }

  proto::Brochure parsed_message;
  {
    Consumer c{rb};

    ssize_t ret = c.startTx();
    EXPECT_EQ(ret, 0);

    ZeroCopyInputStream is(&c, len);
    ASSERT_TRUE(parsed_message.ParseFromZeroCopyStream(&is));

    ret = c.commitTx();
    EXPECT_EQ(ret, 0);

    EXPECT_EQ(rb.getHeader().usedSizeWeak(), 0);
  }

  ASSERT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      message, parsed_message));
}

} // namespace

TEST(RingBuffer, NonWrappingZeroCopyStream) {
  constexpr size_t kBufferSize = 2 * 1024 * 1024;
  RingBufferStorage storage(kBufferSize);
  RingBuffer rb = storage.getRb();

  proto::Brochure message = getMessage();

  test_serialize_parse(rb, message);
}

TEST(RingBuffer, WrappingZeroCopyStream) {
  constexpr size_t kBufferSize = 2 * 1024 * 1024;
  RingBufferStorage storage(kBufferSize);
  RingBuffer rb = storage.getRb();

  // Write and consume enough bytes to force the next write to wrap around.
  {
    Producer p{rb};
    std::array<char, kBufferSize - 1> garbage;
    EXPECT_EQ(p.write(garbage.data(), sizeof(garbage)), sizeof(garbage));
  }
  {
    Consumer c{rb};
    std::array<char, kBufferSize - 1> buf;
    EXPECT_EQ(c.copy(sizeof(buf), buf.data()), sizeof(buf));
  }

  proto::Brochure message = getMessage();
  test_serialize_parse(rb, message);
}
