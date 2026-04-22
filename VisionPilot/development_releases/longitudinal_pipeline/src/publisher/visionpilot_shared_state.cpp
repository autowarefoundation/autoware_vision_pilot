/**
 * @file visionpilot_shared_state.cpp
 * @brief POSIX shared-memory IPC implementation for VisionPilot v0.9.
 */

#include "publisher/visionpilot_shared_state.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace autoware_pov::vision::publisher {

// ---------------------------------------------------------------------------
// Seqlock helpers — use GCC/Clang builtins; safe across process boundaries
// unlike std::atomic which requires process-shared mutexes.
// ---------------------------------------------------------------------------
static inline uint64_t seq_load(const volatile uint64_t* p) {
    uint64_t v;
    __atomic_load(p, &v, __ATOMIC_ACQUIRE);
    return v;
}

static inline void seq_store(volatile uint64_t* p, uint64_t v) {
    __atomic_store(p, &v, __ATOMIC_RELEASE);
}

// Full memory barrier to prevent reordering across the protected region.
static inline void mem_barrier() {
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
VisionPilotSharedState::VisionPilotSharedState(const char* name, bool owner)
    : name_(name), fd_(-1), ptr_(MAP_FAILED), owner_(owner)
{
    int flags = owner ? (O_CREAT | O_RDWR) : O_RDWR;
    int prot  = owner ? (PROT_READ | PROT_WRITE) : PROT_READ;

    fd_ = shm_open(name_, flags, 0666);
    if (fd_ < 0) {
        perror("VisionPilotSharedState: shm_open");
        throw std::runtime_error(std::string("shm_open failed for ") + name_);
    }

    if (owner) {
        if (ftruncate(fd_, sizeof(VisionPilotState)) < 0) {
            perror("VisionPilotSharedState: ftruncate");
            close(fd_);
            throw std::runtime_error("ftruncate failed");
        }
    }

    ptr_ = mmap(nullptr, sizeof(VisionPilotState), prot, MAP_SHARED, fd_, 0);
    if (ptr_ == MAP_FAILED) {
        perror("VisionPilotSharedState: mmap");
        close(fd_);
        throw std::runtime_error("mmap failed");
    }

    if (owner) {
        std::memset(ptr_, 0, sizeof(VisionPilotState));
    }

    std::cout << "[IPC] Shared memory " << (owner ? "created" : "attached")
              << ": " << name_ << " (" << sizeof(VisionPilotState) << " bytes)\n";
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
VisionPilotSharedState::~VisionPilotSharedState() {
    if (ptr_ != MAP_FAILED) {
        munmap(ptr_, sizeof(VisionPilotState));
    }
    if (fd_ >= 0) {
        close(fd_);
    }
    // Only the owner (inference process) unlinks the segment on shutdown.
    if (owner_) {
        shm_unlink(name_);
    }
}

// ---------------------------------------------------------------------------
// publish() — seqlock write
// ---------------------------------------------------------------------------
void VisionPilotSharedState::publish(const VisionPilotState& state) {
    VisionPilotState* s = raw();
    if (!s) return;

    // Step 1: increment seq to ODD → signals "write in progress" to readers
    uint64_t cur_seq = seq_load(&s->seq);
    seq_store(&s->seq, cur_seq + 1);
    mem_barrier();

    // Step 2: copy all fields except seq (don't overwrite the seqlock counter)
    std::memcpy(reinterpret_cast<char*>(s) + sizeof(uint64_t),
                reinterpret_cast<const char*>(&state) + sizeof(uint64_t),
                sizeof(VisionPilotState) - sizeof(uint64_t));

    // Step 3: increment seq to EVEN → write complete
    mem_barrier();
    seq_store(&s->seq, cur_seq + 2);
}

// ---------------------------------------------------------------------------
// read() — seqlock retry loop
// ---------------------------------------------------------------------------
void VisionPilotSharedState::read(VisionPilotState& out) const {
    const VisionPilotState* s = raw();
    if (!s) return;

    uint64_t seq1, seq2;
    do {
        seq1 = seq_load(&s->seq);
        if (seq1 & 1u) continue;  // ODD → write in progress, spin

        mem_barrier();
        std::memcpy(&out, s, sizeof(VisionPilotState));
        mem_barrier();

        seq2 = seq_load(&s->seq);
    } while (seq1 != seq2);  // seq changed mid-copy → retry
}

// ---------------------------------------------------------------------------
// raw()
// ---------------------------------------------------------------------------
VisionPilotState* VisionPilotSharedState::raw() const {
    return reinterpret_cast<VisionPilotState*>(ptr_);
}

} // namespace autoware_pov::vision::publisher
