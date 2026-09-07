// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// RUN: ttlang-opt --ttl-lower-signpost-to-emitc %s | FileCheck %s

// User regions retain their endpoints when values cross the region boundary.
// CHECK-LABEL: func.func @escaping
// CHECK: DeviceZoneBeginN
// CHECK: arith.constant 7
// CHECK: DeviceZoneEnd
// CHECK: return
func.func @escaping() -> i32 {
  ttl.signpost "ttl_layer"
  %value = arith.constant 7 : i32
  ttl.signpost "ttl_layer" {is_end}
  return %value : i32
}
