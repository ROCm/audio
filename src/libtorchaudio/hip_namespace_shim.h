#pragma once

#include <libtorchaudio/hip_utils.h>

namespace libtorchaudio::hip {

inline hipStream_t getCurrentHIPStreamMasqueradingAsCUDA(
    torch::stable::DeviceIndex device_index = -1) {
  return cuda::getCurrentHIPStreamMasqueradingAsCUDA(device_index);
}

inline void setCurrentHIPStreamMasqueradingAsCUDA(
    hipStream_t stream,
    torch::stable::DeviceIndex device_index = -1) {
  cuda::setCurrentHIPStreamMasqueradingAsCUDA(stream, device_index);
}

inline hipStream_t getStreamFromPoolMasqueradingAsCUDA(
    const bool isHighPriority = false,
    torch::stable::DeviceIndex device_index = -1) {
  return cuda::getStreamFromPoolMasqueradingAsCUDA(isHighPriority, device_index);
}

inline void synchronize(
    hipStream_t stream,
    torch::stable::DeviceIndex device_index = -1) {
  cuda::synchronize(stream, device_index);
}

} // namespace libtorchaudio::hip
