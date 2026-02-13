# Makefile for Navier-Stokes Solver using deal.II

# Default build directory
BUILD_DIR = build

.PHONY: all clean run setup

# Default target: build the project
all: $(BUILD_DIR)/Makefile
	@echo "Building project..."
	@$(MAKE) -C $(BUILD_DIR)

# Run cmake to generate the Makefile in the build directory
$(BUILD_DIR)/Makefile: CMakeLists.txt
	@echo "Running CMake..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Release ..

# Convenience target to run the executable
run: all
	@echo "Running simulation..."
	@cd $(BUILD_DIR) && ./navier_stokes

# Clean the build directory
clean:
	@echo "Cleaning up..."
	@rm -rf $(BUILD_DIR)

# Direct build if needed
setup: clean $(BUILD_DIR)/Makefile
