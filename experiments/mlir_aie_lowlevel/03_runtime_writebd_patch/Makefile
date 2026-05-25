PYTHON ?= python3
BUILD_DIR ?= build
MLIR ?= $(BUILD_DIR)/runtime_writebd_patch.mlir
XCLBIN ?= $(BUILD_DIR)/runtime_writebd_patch.xclbin
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

