//
// Created by atanasko on 2/12/26.
//
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include "publisher/ipc_shared_state.hpp"

namespace visionpilot::publisher
{
IpcSharedState::IpcSharedState(const char* name) : name_(name) {
  fd_ = shm_open(name_, O_CREAT | O_RDWR, 0666);
  if (fd_ < 0) {
    perror("shm_open");
    throw std::runtime_error("shm_open failed");
  }

  ftruncate(fd_, sizeof(ControlState));

  ptr_ = mmap(nullptr, sizeof(ControlState),
              PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);

  if (ptr_ == MAP_FAILED) {
    perror("mmap");
    throw std::runtime_error("mmap failed");
  }

  std::memset(ptr_, 0, sizeof(ControlState));
}

IpcSharedState::~IpcSharedState() {
  munmap(ptr_, sizeof(ControlState));
  close(fd_);
  // shm_unlink(name_); // Only unlink in creator process if needed
}

ControlState* IpcSharedState::get() {
  return reinterpret_cast<ControlState*>(ptr_);
}

void IpcSharedState::set(double steering, double velocity) {
  ControlState* state = get();
  if (!state) return;

  // Start write: mark seq as odd
  // uint64_t s = state->seq.load(std::memory_order_relaxed);
  // state->seq.store(s + 1, std::memory_order_release);

  // Write values
  state->steering = steering;
  state->velocity = velocity;

  // Finish write: mark seq as even
  // state->seq.store(s + 2, std::memory_order_release);
}
} // namespace visionpilot::publisher