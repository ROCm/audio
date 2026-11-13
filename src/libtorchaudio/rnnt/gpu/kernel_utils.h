#pragma once

#include <cassert>

#include <libtorchaudio/rnnt/gpu/math.cuh>

namespace torchaudio {
namespace rnnt {

inline HOST_AND_DEVICE bool in_range(
    int start,
    int end, // inclusive
    int val) {
  return start <= val && val <= end;
}

#define LOG_PROBS_SKIP_IDX 0
#define LOG_PROBS_EMIT_IDX 1

// Sizes are stored by value. A reference member dangles when the indexer is
// built from a temporary, such as Indexer2D(maxU - 1), which the gradient
// kernel keeps and uses after that temporary is gone.
struct Indexer2D {
  int size2_;

  FORCE_INLINE HOST_AND_DEVICE Indexer2D(int size2) : size2_(size2) {}

  FORCE_INLINE HOST_AND_DEVICE int operator()(int index1, int index2) const {
    return index1 * size2_ + index2;
  }
};

struct Indexer3D {
  int size2_;
  int size3_;

  FORCE_INLINE HOST_AND_DEVICE Indexer3D(int size2, int size3)
      : size2_(size2), size3_(size3) {}

  FORCE_INLINE HOST_AND_DEVICE int operator()(
      int index1,
      int index2,
      int index3) const {
    return (index1 * size2_ + index2) * size3_ + index3;
  }
};

struct Indexer4D {
  int size2_;
  int size3_;
  int size4_;

  HOST_AND_DEVICE Indexer4D(int size2, int size3, int size4)
      : size2_(size2), size3_(size3), size4_(size4) {}

  HOST_AND_DEVICE int operator()(
      int index1,
      int index2,
      int index3,
      int index4) const {
    return ((index1 * size2_ + index2) * size3_ + index3) * size4_ + index4;
  }
};

} // namespace rnnt
} // namespace torchaudio
