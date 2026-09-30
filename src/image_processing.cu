#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kBlockSize = 256;

struct Image {
  int width = 0;
  int height = 0;
  std::vector<unsigned char> pixels;
};

void CheckCuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA error during " << operation << ": "
              << cudaGetErrorString(error) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

Image ReadPgm(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Cannot open input image: " + path.string());
  }

  std::string magic;
  input >> magic;
  if (magic != "P5") {
    throw std::runtime_error("Only binary PGM (P5) files are supported: " +
                             path.string());
  }

  int width = 0;
  int height = 0;
  int max_value = 0;
  input >> width >> height >> max_value;
  input.get();

  if (width <= 0 || height <= 0 || max_value != 255) {
    throw std::runtime_error("Invalid PGM header: " + path.string());
  }

  Image image;
  image.width = width;
  image.height = height;
  image.pixels.resize(static_cast<size_t>(width) * height);
  input.read(reinterpret_cast<char*>(image.pixels.data()),
             static_cast<std::streamsize>(image.pixels.size()));

  if (input.gcount() != static_cast<std::streamsize>(image.pixels.size())) {
    throw std::runtime_error("Incomplete image data: " + path.string());
  }
  return image;
}

void WritePgm(const fs::path& path, const Image& image) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("Cannot create output image: " + path.string());
  }

  output << "P5\n" << image.width << ' ' << image.height << "\n255\n";
  output.write(reinterpret_cast<const char*>(image.pixels.data()),
               static_cast<std::streamsize>(image.pixels.size()));
}

__global__ void EnhanceImageKernel(const unsigned char* input,
                                   unsigned char* output,
                                   size_t pixel_count,
                                   float contrast,
                                   int brightness) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (index < pixel_count) {
    const float centered =
        (static_cast<float>(input[index]) - 128.0f) * contrast + 128.0f +
        static_cast<float>(brightness);
    const float clamped = fminf(255.0f, fmaxf(0.0f, centered));
    output[index] = static_cast<unsigned char>(clamped);
  }
}

struct Options {
  fs::path input_dir = "data/input";
  fs::path output_dir = "data/output";
  float contrast = 1.20f;
  int brightness = 5;
  int threads = kBlockSize;
};

void PrintUsage(const char* program) {
  std::cout
      << "Usage: " << program
      << " [--input DIR] [--output DIR] [--contrast VALUE]\n"
      << "       [--brightness INT] [--threads INT]\n\n"
      << "Example:\n"
      << "  " << program
      << " --input data/input --output data/output --threads 256\n";
}

bool StartsWith(const std::string& value, const std::string& prefix) {
  return value.rfind(prefix, 0) == 0;
}

Options ParseArguments(int argc, char* argv[]) {
  Options options;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];

    if (argument == "--help" || argument == "-h") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }

    auto require_value = [&](const std::string& flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + flag);
      }
      return argv[++i];
    };

    if (argument == "--input") {
      options.input_dir = require_value(argument);
    } else if (argument == "--output") {
      options.output_dir = require_value(argument);
    } else if (argument == "--contrast") {
      options.contrast = std::stof(require_value(argument));
    } else if (argument == "--brightness") {
      options.brightness = std::stoi(require_value(argument));
    } else if (argument == "--threads") {
      options.threads = std::stoi(require_value(argument));
    } else if (StartsWith(argument, "--")) {
      throw std::runtime_error("Unknown argument: " + argument);
    } else {
      throw std::runtime_error("Unexpected argument: " + argument);
    }
  }

  if (options.threads <= 0 || options.threads > 1024) {
    throw std::runtime_error("--threads must be between 1 and 1024");
  }
  if (options.contrast < 0.0f) {
    throw std::runtime_error("--contrast must be non-negative");
  }

  return options;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const Options options = ParseArguments(argc, argv);

    fs::create_directories(options.output_dir);

    std::vector<fs::path> input_files;
    for (const auto& entry : fs::directory_iterator(options.input_dir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".pgm") {
        input_files.push_back(entry.path());
      }
    }
    std::sort(input_files.begin(), input_files.end());

    if (input_files.empty()) {
      throw std::runtime_error("No .pgm files found in " +
                               options.input_dir.string());
    }

    int device_count = 0;
    CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("No CUDA-capable GPU was detected.");
    }

    cudaDeviceProp device_properties{};
    CheckCuda(cudaGetDeviceProperties(&device_properties, 0),
              "cudaGetDeviceProperties");

    std::cout << "CUDA Batch Image Processing\n"
              << "===========================\n"
              << "Input directory : " << options.input_dir << '\n'
              << "Output directory: " << options.output_dir << '\n'
              << "Images found    : " << input_files.size() << '\n'
              << "GPU             : " << device_properties.name << '\n'
              << "CUDA threads    : " << options.threads << '\n'
              << "Contrast        : " << options.contrast << '\n'
              << "Brightness      : " << options.brightness << "\n\n";

    cudaEvent_t start;
    cudaEvent_t stop;
    CheckCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    CheckCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    double total_cpu_ms = 0.0;
    float total_gpu_ms = 0.0f;
    size_t processed = 0;

    for (const fs::path& input_path : input_files) {
      const auto cpu_start = std::chrono::steady_clock::now();

      Image input = ReadPgm(input_path);
      Image output;
      output.width = input.width;
      output.height = input.height;
      output.pixels.resize(input.pixels.size());

      unsigned char* device_input = nullptr;
      unsigned char* device_output = nullptr;
      const size_t bytes = input.pixels.size() * sizeof(unsigned char);

      CheckCuda(cudaMalloc(&device_input, bytes), "cudaMalloc(input)");
      CheckCuda(cudaMalloc(&device_output, bytes), "cudaMalloc(output)");

      CheckCuda(cudaMemcpy(device_input, input.pixels.data(), bytes,
                           cudaMemcpyHostToDevice),
                "cudaMemcpy(H2D)");

      const int blocks =
          static_cast<int>((input.pixels.size() + options.threads - 1) /
                           options.threads);

      CheckCuda(cudaEventRecord(start), "cudaEventRecord(start)");
      EnhanceImageKernel<<<blocks, options.threads>>>(
          device_input, device_output, input.pixels.size(), options.contrast,
          options.brightness);
      CheckCuda(cudaGetLastError(), "kernel launch");
      CheckCuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
      CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

      float gpu_ms = 0.0f;
      CheckCuda(cudaEventElapsedTime(&gpu_ms, start, stop),
                "cudaEventElapsedTime");
      total_gpu_ms += gpu_ms;

      CheckCuda(cudaMemcpy(output.pixels.data(), device_output, bytes,
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy(D2H)");

      CheckCuda(cudaFree(device_input), "cudaFree(input)");
      CheckCuda(cudaFree(device_output), "cudaFree(output)");

      const fs::path output_path = options.output_dir / input_path.filename();
      WritePgm(output_path, output);

      const auto cpu_end = std::chrono::steady_clock::now();
      total_cpu_ms += std::chrono::duration<double, std::milli>(
                          cpu_end - cpu_start)
                          .count();

      ++processed;
      std::cout << "Processed " << processed << "/" << input_files.size()
                << ": " << input_path.filename() << " | GPU kernel: "
                << std::fixed << std::setprecision(3) << gpu_ms << " ms\n";
    }

    CheckCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    CheckCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

    std::cout << "\nProcessing completed successfully.\n"
              << "Images processed : " << processed << '\n'
              << "Total GPU kernel time: " << std::fixed
              << std::setprecision(3) << total_gpu_ms << " ms\n"
              << "Total wall-clock time: " << std::fixed
              << std::setprecision(3) << total_cpu_ms << " ms\n";

    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
