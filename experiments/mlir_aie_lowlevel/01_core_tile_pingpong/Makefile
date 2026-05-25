PYTHON ?= python3
BUILD_DIR ?= build
MLIR ?= $(BUILD_DIR)/core_tile_pingpong.mlir
XCLBIN ?= $(BUILD_DIR)/core_tile_pingpong.xclbin
KERNEL ?= MLIR_AIE

.PHONY: all generate check compile clean

all: check

generate:
	mkdir -p "$(BUILD_DIR)"
	"$(PYTHON)" generate.py --output "$(MLIR)"

check: generate
	"$(PYTHON)" check_mlir.py "$(MLIR)"

compile: check
	"$(PYTHON)" compile.py "$(MLIR)" --xclbin "$(XCLBIN)" --kernel "$(KERNEL)"

clean:
	rm -rf "$(BUILD_DIR)"

