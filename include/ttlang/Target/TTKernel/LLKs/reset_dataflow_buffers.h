// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttlang/Target/TTKernel/LLKs/risc_barrier.h"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) ||               \
    defined(COMPILE_FOR_DM)
#include "api/dataflow/dataflow_api.h"
#elif defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_MATH) ||                  \
    defined(UCK_CHLKC_PACK) || defined(TRISC_UNPACK) || defined(TRISC_MATH) || \
    defined(TRISC_PACK)
#include "api/compute/common.h"
#endif

#if !defined(ARCH_BLACKHOLE)
#error "reset_dataflow_buffers currently supports Blackhole only"
#endif

namespace ttlang {
namespace detail {

static constexpr uint32_t kConfigWords = 11;
static constexpr uint32_t kCompletionMarker = 0xd1fb;

#if defined(COMPILE_FOR_BRISC) ||                                              \
    (defined(COMPILE_FOR_DM) && COMPILE_FOR_DM == 0)
#define TTLANG_DFB_DM0
#endif

#if defined(COMPILE_FOR_NCRISC) ||                                             \
    (defined(COMPILE_FOR_DM) && COMPILE_FOR_DM == 1)
#define TTLANG_DFB_DM1
#endif

#if defined(UCK_CHLKC_UNPACK) || defined(TRISC_UNPACK)
#define TTLANG_DFB_UNPACK
#endif

#if defined(UCK_CHLKC_MATH) || defined(TRISC_MATH)
#define TTLANG_DFB_MATH
#endif

#if defined(UCK_CHLKC_PACK) || defined(TRISC_PACK)
#define TTLANG_DFB_PACK
#endif

FORCE_INLINE void drain() {
#if defined(TTLANG_DFB_DM0) || defined(TTLANG_DFB_DM1)
  noc_async_full_barrier();
#elif defined(TTLANG_DFB_UNPACK)
  TTI_STALLWAIT(p_stall::STALL_TDMA, p_stall::UNPACK);
  TTI_SETDMAREG(0, kCompletionMarker, 0, LO_16(p_gpr_unpack::TMP0));
  sync_regfile_write(p_gpr_unpack::TMP0);
#elif defined(TTLANG_DFB_MATH)
  TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
  tensix_sync();
#elif defined(TTLANG_DFB_PACK)
  TTI_STALLWAIT(p_stall::STALL_TDMA, p_stall::PACK);
  TTI_SETDMAREG(0, kCompletionMarker, 0, LO_16(p_gpr_pack::TMP0));
  sync_regfile_write(p_gpr_pack::TMP0);
#endif
}

FORCE_INLINE void reconfigureInterface(uint32_t id, uint32_t totalBytes,
                                       uint32_t numPages, uint32_t pageBytes) {
#if defined(TTLANG_DFB_DM0) || defined(TTLANG_DFB_DM1) ||                      \
    defined(TTLANG_DFB_UNPACK) || defined(TTLANG_DFB_PACK)
  LocalCBInterface &iface = get_local_cb_interface(id);
  const uint32_t base = iface.fifo_limit - iface.fifo_size;
  const uint32_t size = totalBytes >> cb_addr_shift;
  const uint32_t pageSize = pageBytes >> cb_addr_shift;

#if defined(TTLANG_DFB_DM0) || defined(TTLANG_DFB_DM1)
  iface.fifo_rd_ptr = base;
  iface.fifo_wr_ptr = base;
  iface.fifo_num_pages = numPages;
#elif defined(TTLANG_DFB_UNPACK)
  iface.fifo_rd_ptr = base;
#elif defined(TTLANG_DFB_PACK)
  iface.fifo_wr_ptr = base;
  iface.fifo_wr_tile_ptr = 0;
  iface.fifo_num_pages = numPages;
#endif
  iface.fifo_size = size;
  iface.fifo_limit = base + size;
  iface.fifo_page_size = pageSize;
  iface.tiles_acked_received_init = 0;

#if defined(TTLANG_DFB_DM1)
  *get_cb_tiles_received_ptr(id) = 0;
  *get_cb_tiles_acked_ptr(id) = 0;
#endif
#endif
}

FORCE_INLINE void reconfigureFormat(uint32_t id, uint32_t pageBytes,
                                    uint32_t l1Format, uint32_t tileHeight,
                                    uint32_t tileWidth, uint32_t faceHeight,
                                    uint32_t numFaces, uint32_t unpackDstFormat,
                                    uint32_t packSrcFormat) {
#if defined(TTLANG_DFB_DM0) || defined(TTLANG_DFB_DM1) ||                      \
    defined(TTLANG_DFB_UNPACK) || defined(TTLANG_DFB_MATH)
  unpack_src_format[id] = l1Format;
  unpack_dst_format[id] = unpackDstFormat;
  unpack_tile_num_faces[id] = numFaces;
  unpack_partial_face[id] = tileHeight < 32;
  unpack_tile_face_r_dim[id] = faceHeight;
  unpack_narrow_tile[id] = tileWidth < 32;
  unpack_tile_r_dim[id] = tileHeight;
  unpack_tile_c_dim[id] = tileWidth;
  unpack_tile_size[id] = pageBytes;
  unpack_num_faces_c_dim[id] =
      numFaces < tileWidth / 16 ? numFaces : tileWidth / 16;
  unpack_num_faces_r_dim[id] = numFaces / unpack_num_faces_c_dim[id];
#endif

#if defined(TTLANG_DFB_DM0) || defined(TTLANG_DFB_DM1) ||                      \
    defined(TTLANG_DFB_PACK)
  pack_src_format[id] = packSrcFormat;
  pack_dst_format[id] = l1Format;
#if defined(TTLANG_DFB_PACK)
  unpack_src_format[id] = l1Format;
#endif
  pack_tile_num_faces[id] = numFaces;
  pack_partial_face[id] = tileHeight < 32;
  pack_tile_face_r_dim[id] = faceHeight;
  pack_narrow_tile[id] = tileWidth < 32;
  pack_tile_r_dim[id] = tileHeight;
  pack_tile_c_dim[id] = tileWidth;
  pack_tile_size[id] = pageBytes;
  pack_num_faces_c_dim[id] =
      numFaces < tileWidth / 16 ? numFaces : tileWidth / 16;
  pack_num_faces_r_dim[id] = numFaces / pack_num_faces_c_dim[id];
#endif
}

// Immediate arguments avoid duplicating reset instructions or filling local
// RAM.
__attribute__((noinline, noclone)) static void
applyConfiguration(uint32_t numPages, uint32_t geometry, uint32_t formats) {
  const uint32_t pageBytes = geometry & 0xffff;
  const uint32_t id = (geometry >> 16) & 0x3f;
  const uint32_t numFaces = ((geometry >> 22) & 3) + 1;
  const uint32_t l1Format = geometry >> 24;
  const uint32_t tileHeight = formats & 0xff;
  const uint32_t tileWidth = (formats >> 8) & 0xff;
  const uint32_t unpackDstFormat = (formats >> 16) & 0xff;
  const uint32_t packSrcFormat = formats >> 24;
  reconfigureInterface(id, numPages * pageBytes, numPages, pageBytes);
  reconfigureFormat(id, pageBytes, l1Format, tileHeight, tileWidth,
                    tileHeight < 16 ? tileHeight : 16, numFaces,
                    unpackDstFormat, packSrcFormat);
}

template <uint32_t... Config>
struct ApplyConfigurations;

template <>
struct ApplyConfigurations<> {
  static FORCE_INLINE void run() {}
};

template <uint32_t Id, uint32_t TotalBytes, uint32_t NumPages,
          uint32_t PageBytes, uint32_t L1Format, uint32_t TileHeight,
          uint32_t TileWidth, uint32_t FaceHeight, uint32_t NumFaces,
          uint32_t UnpackDstFormat, uint32_t PackSrcFormat,
          uint32_t... Remaining>
struct ApplyConfigurations<Id, TotalBytes, NumPages, PageBytes, L1Format,
                           TileHeight, TileWidth, FaceHeight, NumFaces,
                           UnpackDstFormat, PackSrcFormat, Remaining...> {
  static FORCE_INLINE void run() {
    static_assert(Id < 64 && PageBytes <= UINT16_MAX);
    static_assert(NumFaces >= 1 && NumFaces <= 4);
    static_assert(L1Format <= UINT8_MAX && TileHeight <= UINT8_MAX &&
                  TileWidth <= UINT8_MAX && UnpackDstFormat <= UINT8_MAX &&
                  PackSrcFormat <= UINT8_MAX);
    static_assert(TotalBytes == NumPages * PageBytes);
    static_assert(FaceHeight == (TileHeight < 16 ? TileHeight : 16));
    constexpr uint32_t geometry =
        PageBytes | (Id << 16) | ((NumFaces - 1) << 22) | (L1Format << 24);
    constexpr uint32_t formats = TileHeight | (TileWidth << 8) |
                                 (UnpackDstFormat << 16) |
                                 (PackSrcFormat << 24);
    applyConfiguration(NumPages, geometry, formats);
    ApplyConfigurations<Remaining...>::run();
  }
};

} // namespace detail

template <uint32_t RecordCount, uint32_t... Config>
__attribute__((noinline)) static void
reset_dataflow_buffers(uint32_t word0, uint32_t word1, uint32_t word2,
                       uint32_t word3) {
  static_assert(sizeof...(Config) == RecordCount * detail::kConfigWords);

  detail::drain();
  detail::riscBarrierEnter(word0, word1, word2, word3);
  detail::ApplyConfigurations<Config...>::run();
  asm volatile("" ::: "memory");
  detail::riscBarrierExit(word0, word1, word2, word3);
}

} // namespace ttlang

#undef TTLANG_DFB_DM0
#undef TTLANG_DFB_DM1
#undef TTLANG_DFB_UNPACK
#undef TTLANG_DFB_MATH
#undef TTLANG_DFB_PACK
