EXPERIMENTS := 01_core_tile_pingpong 02_memtile_kv_plane_ring 03_runtime_writebd_patch

.PHONY: all check compile clean $(EXPERIMENTS)

all: check

check: $(EXPERIMENTS)

$(EXPERIMENTS):
	$(MAKE) -C "$@"

compile:
	for d in $(EXPERIMENTS); do $(MAKE) -C "$$d" compile; done

clean:
	for d in $(EXPERIMENTS); do $(MAKE) -C "$$d" clean; rm -rf "$$d"/*.mlir.prj; done

