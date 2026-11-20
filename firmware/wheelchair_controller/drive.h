// What the motors are allowed to do: ramps, the command watchdog and the
// emergency stop, as one small state machine with the clock passed in.
//
// The host asks for wheel speeds several times a second. This class turns the
// requests into outputs that never jump, and takes the outputs to zero when
//
//   * no valid drive command has arrived for `watchdogMs` (the host program
//     crashed, the cable came out, the laptop went to sleep), or
//   * a stop is commanded, or
//   * the emergency-stop input is active. That one latches: the outputs stay at
//     zero until the input has been released and the host sends a release.
//
// No Arduino dependencies: the same file is compiled into the host-side tests.
#pragma once

#include <stdint.h>

namespace nwc {

enum class DriveState : char { Idle = 'I', Driving = 'D', Watchdog = 'W', EmergencyStop = 'E' };

struct DriveConfig {
  uint16_t watchdogMs;
  uint16_t rampUpMs;    // time from standstill to full speed
  uint16_t rampDownMs;  // time from full speed to standstill

  // A constructor, not member initialisers: the Arduino toolchain builds as C++11.
  explicit DriveConfig(uint16_t watchdog = 400, uint16_t rampUp = 600, uint16_t rampDown = 200)
      : watchdogMs(watchdog), rampUpMs(rampUp), rampDownMs(rampDown) {}
};

class Drive {
 public:
  explicit Drive(const DriveConfig& config = DriveConfig()) : config_(config) {}

  // A checked drive command arrived at time `nowMs`.
  void command(int16_t left, int16_t right, uint32_t nowMs) {
    lastCommandMs_ = nowMs;
    everCommanded_ = true;
    if (state_ == DriveState::EmergencyStop) return;
    targetLeft_ = clamp(left);
    targetRight_ = clamp(right);
    if (state_ == DriveState::Watchdog) state_ = DriveState::Idle;
  }

  void stop() {
    targetLeft_ = 0;
    targetRight_ = 0;
  }

  // Level of the emergency-stop input. Active cuts the outputs at once and latches.
  void emergencyInput(bool active) {
    inputActive_ = active;
    if (active) {
      state_ = DriveState::EmergencyStop;
      targetLeft_ = targetRight_ = 0;
      left_ = right_ = 0;
    }
  }

  // The host asks to leave the emergency stop. Refused while the input is still active.
  bool release() {
    if (state_ != DriveState::EmergencyStop || inputActive_) return false;
    state_ = DriveState::Idle;
    return true;
  }

  // Advance to `nowMs`. Call often (every few milliseconds).
  void update(uint32_t nowMs) {
    const uint32_t elapsed = started_ ? nowMs - lastUpdateMs_ : 0;
    lastUpdateMs_ = nowMs;
    started_ = true;
    if (state_ != DriveState::EmergencyStop) {
      if (everCommanded_ && nowMs - lastCommandMs_ > config_.watchdogMs) {
        targetLeft_ = targetRight_ = 0;
        state_ = DriveState::Watchdog;
      }
      left_ = approach(left_, targetLeft_, elapsed);
      right_ = approach(right_, targetRight_, elapsed);
      if (state_ != DriveState::Watchdog) {
        state_ = (left_ != 0 || right_ != 0 || targetLeft_ != 0 || targetRight_ != 0) ? DriveState::Driving : DriveState::Idle;
      }
    }
  }

  // Outputs, -255..255.
  int16_t left() const { return static_cast<int16_t>(left_ / SCALE); }
  int16_t right() const { return static_cast<int16_t>(right_ / SCALE); }
  DriveState state() const { return state_; }

  // Milliseconds since the last valid drive command, capped at 9999.
  uint16_t commandAge(uint32_t nowMs) const {
    if (!everCommanded_) return 9999;
    const uint32_t age = nowMs - lastCommandMs_;
    return age > 9999 ? 9999 : static_cast<uint16_t>(age);
  }

 private:
  static constexpr int32_t SCALE = 256;  // outputs are kept in 1/256 steps so that slow ramps do not stall

  static int32_t clamp(int16_t value) {
    const int32_t v = value > 255 ? 255 : (value < -255 ? -255 : value);
    return v * SCALE;
  }

  // Move `current` towards `target`. Moving towards zero uses the fast ramp, away from it the slow one;
  // a change of direction first comes to a stop.
  int32_t approach(int32_t current, int32_t target, uint32_t elapsedMs) const {
    if (current == target) return current;
    const bool sameSign = (current >= 0 && target >= 0) || (current <= 0 && target <= 0);
    const int32_t goal = sameSign ? target : 0;
    const int32_t magnitude = current < 0 ? -current : current;
    const int32_t goalMagnitude = goal < 0 ? -goal : goal;
    const bool slowing = goalMagnitude < magnitude;
    const uint16_t rampMs = slowing ? config_.rampDownMs : config_.rampUpMs;
    const int32_t step = rampMs == 0 ? 255 * SCALE : static_cast<int32_t>((255L * SCALE * static_cast<int32_t>(elapsedMs > 1000 ? 1000 : elapsedMs)) / rampMs);
    if (goal > current) return (goal - current <= step) ? goal : current + step;
    return (current - goal <= step) ? goal : current - step;
  }

  DriveConfig config_;
  DriveState state_ = DriveState::Idle;
  int32_t left_ = 0, right_ = 0, targetLeft_ = 0, targetRight_ = 0;
  uint32_t lastCommandMs_ = 0, lastUpdateMs_ = 0;
  bool everCommanded_ = false, started_ = false, inputActive_ = false;
};

}  // namespace nwc
