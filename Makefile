CXX := nvcc
TARGET := cuda_image_processor
SOURCE := src/image_processing.cu
CXXFLAGS := -O2 -std=c++17

.PHONY: all clean dataset run validate

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET)

dataset:
	python3 scripts/generate_dataset.py

run: $(TARGET)
	./$(TARGET) --input data/input --output data/output --threads 256

validate:
	python3 scripts/validate_output.py data/input data/output

clean:
	rm -f $(TARGET)
	rm -f data/output/*.pgm
