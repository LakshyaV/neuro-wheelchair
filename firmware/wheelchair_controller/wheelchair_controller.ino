// Motor controller for the EEG-driven wheelchair.
//
// The host sends wheel speeds over USB serial several times a second (see
// protocol.h). The controller applies them to an L298N H-bridge with ramps, and
// stops the motors by itself when the commands stop coming, when a stop is
// commanded, or when the emergency-stop input opens (see drive.h).
//
// Build-time settings are in config.h. The code is in controller.h.

#include "controller.h"

void setup() { nwc::controller::setup(); }

void loop() { nwc::controller::loop(); }
