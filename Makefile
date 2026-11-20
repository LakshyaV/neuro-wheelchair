.PHONY: install lint format test check firmware config demo

install:
	pip install -e ".[all]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest

check: lint test

# Compile the motor controller for an Arduino Uno in its three wiring variants.
firmware:
	arduino-cli compile --fqbn arduino:avr:uno --warnings all firmware/wheelchair_controller
	arduino-cli compile --fqbn arduino:avr:uno --warnings all --build-property "compiler.cpp.extra_flags=-DNWC_PWM_ON_ENABLE=1" firmware/wheelchair_controller
	arduino-cli compile --fqbn arduino:avr:uno --warnings all --build-property "compiler.cpp.extra_flags=-DNWC_ESTOP_PIN=7 -DNWC_MIN_DUTY=60" firmware/wheelchair_controller

# Regenerate configs/default.yaml after changing a settings dataclass.
config:
	python tools/write_default_config.py

demo:
	nwc demo
