TEST_DIR = TrainingWeights
PY_GEN = wnb.py

TARGET = ChessNN

$(TARGET): main.cpp
	g++ -O3 -march=native main.cpp -lopenblas -o $(TARGET) -ffast-math -flto

debug:
	g++ main.cpp -o ChessNNDebug -O0 -g -march=native -Wextra -Wall -lopenblas
	gdb ChessNNDebug

.PHONY: testdata
testdata:
	@echo "Running Python generator..."
	# 1. Run the python script (assumes it outputs files in current dir)
	python3 $(PY_GEN)
	# 2. Ensure test directory exists
	mkdir -p $(TEST_DIR)
	# 3. Move the generated files (adjust extension as needed, e.g., *.txt or *.csv)
	mv *.csv $(TEST_DIR)/
	@echo "Test files updated in $(TEST_DIR)."

.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf ChessNN
	rm -rf ChessNNDebug
	# Delete the generated test files directory
	rm -rf $(TEST_DIR)
	# Optionally re-run test generation immediately if that's your workflow
	@$(MAKE) testdata
	@echo "Clean and Re-generation complete."

