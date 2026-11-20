// Serial protocol between the host and the motor controller.
//
// Every message is one line of ASCII:
//
//     $<payload>*<HH>\n
//
// <payload> is a comma-separated list whose first field is the message type and
// <HH> is the CRC-8 (polynomial 0x07) of the payload as two hex digits. A line
// that is too long, malformed or fails its checksum is dropped and counted.
//
// Host to controller
//     D,<seq>,<left>,<right>   drive: wheel speeds, -255..255
//     S,<seq>                  stop
//     R,<seq>                  release a latched emergency stop
//     Q,<seq>                  ask for the information message
//
// Controller to host
//     T,<seq>,<state>,<left>,<right>,<age_ms>,<rejected>   telemetry, 10 per second
//     I,<protocol>,<firmware>,<watchdog_ms>,<ramp_up_ms>,<ramp_down_ms>
//
// This header has no Arduino dependencies, so the parser is compiled and tested
// on the host as well.
#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace nwc {

constexpr uint8_t PROTOCOL_VERSION = 1;
constexpr size_t MAX_LINE = 48;

inline uint8_t crc8(const char* data, size_t length) {
  uint8_t crc = 0;
  for (size_t i = 0; i < length; i++) {
    crc ^= static_cast<uint8_t>(data[i]);
    for (uint8_t bit = 0; bit < 8; bit++) {
      crc = (crc & 0x80) ? static_cast<uint8_t>((crc << 1) ^ 0x07) : static_cast<uint8_t>(crc << 1);
    }
  }
  return crc;
}

struct Command {
  char type = 0;  // 'D', 'S', 'R' or 'Q'
  uint16_t seq = 0;
  int16_t left = 0;
  int16_t right = 0;
};

inline int hexValue(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;
}

// Reads one integer field: an optional minus sign and one to five digits, nothing else.
// A field that is not the last one must be followed by a comma, the last one by the end.
inline bool readField(const char*& cursor, long low, long high, bool last, long& value) {
  bool negative = false;
  if (*cursor == '-') {
    negative = true;
    cursor++;
  }
  long magnitude = 0;
  uint8_t digits = 0;
  while (*cursor >= '0' && *cursor <= '9') {
    if (++digits > 5) return false;
    magnitude = magnitude * 10 + (*cursor - '0');
    cursor++;
  }
  if (digits == 0) return false;
  value = negative ? -magnitude : magnitude;
  if (value < low || value > high) return false;
  if (last) return *cursor == '\0';
  if (*cursor != ',') return false;
  cursor++;
  return true;
}

// Checks one complete line (without the line ending) and fills `out`.
inline bool parseLine(const char* line, size_t length, Command& out) {
  if (length < 7 || length > MAX_LINE || line[0] != '$' || line[length - 3] != '*') return false;
  const size_t payloadLength = length - 4;
  const int high = hexValue(line[length - 2]);
  const int low = hexValue(line[length - 1]);
  if (high < 0 || low < 0 || ((high << 4) | low) != crc8(line + 1, payloadLength)) return false;

  char payload[MAX_LINE];
  memcpy(payload, line + 1, payloadLength);
  payload[payloadLength] = '\0';
  if (payload[1] != ',') return false;

  const char type = payload[0];
  const char* cursor = payload + 2;
  long seq = 0, left = 0, right = 0;
  if (type == 'D') {
    if (!readField(cursor, 0, 65535, false, seq) || !readField(cursor, -255, 255, false, left) || !readField(cursor, -255, 255, true, right)) return false;
  } else if (type == 'S' || type == 'R' || type == 'Q') {
    if (!readField(cursor, 0, 65535, true, seq)) return false;
  } else {
    return false;
  }
  out = Command();
  out.type = type;
  out.seq = static_cast<uint16_t>(seq);
  out.left = static_cast<int16_t>(left);
  out.right = static_cast<int16_t>(right);
  return true;
}

// Collects bytes into lines. Feed every received byte; when it returns true,
// `command` holds a checked command.
class LineReader {
 public:
  bool feed(char byte, Command& command) {
    if (byte == '\r') return false;
    if (byte == '\n') {
      const bool ok = !overflow_ && length_ > 0 && parseLine(buffer_, length_, command);
      if (!ok && (overflow_ || length_ > 0)) rejected_++;
      length_ = 0;
      overflow_ = false;
      return ok;
    }
    if (length_ < MAX_LINE) {
      buffer_[length_++] = byte;
    } else {
      overflow_ = true;
    }
    return false;
  }

  uint16_t rejected() const { return rejected_; }

 private:
  char buffer_[MAX_LINE] = {0};
  size_t length_ = 0;
  bool overflow_ = false;
  uint16_t rejected_ = 0;
};

// Writes "$<payload>*HH\n" into `out` (at least MAX_LINE + 8 bytes); returns its length.
inline size_t frame(const char* payload, char* out) {
  const size_t length = strlen(payload);
  return static_cast<size_t>(snprintf(out, MAX_LINE + 8, "$%s*%02X\n", payload, crc8(payload, length)));
}

}  // namespace nwc
