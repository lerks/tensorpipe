/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <tensorpipe/test/transport/ib/ib_test.h>

namespace {

IbTransportTestHelper helper;

} // namespace

INSTANTIATE_TEST_CASE_P(Ib, TransportTest, ::testing::Values(&helper));
