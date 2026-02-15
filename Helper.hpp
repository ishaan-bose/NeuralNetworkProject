#include <random>
#include <chrono>
#include <immintrin.h>
#include "rapidcsv.h"
#include <iomanip>

// Function to save weight gradients to CSV files
void save_weight_gradient_to_csv(const std::string& filename, double* gradient, int rows, int cols) {
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(10);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file << gradient[i * cols + j];
            if (j < cols - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
    
    file.close();
}

// Function to save bias gradients to CSV files (single row)
void save_bias_gradient_to_csv(const std::string& filename, double* gradient, int size) {
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(10);
    
    for (int i = 0; i < size; i++) {
        file << gradient[i];
        if (i < size - 1) {
            file << ",";
        }
    }
    file << "\n";
    
    file.close();
}

// Function to save single bias gradient value to CSV
void save_single_bias_gradient_to_csv(const std::string& filename, double gradient) {
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(10);
    file << gradient << "\n";
    file.close();
}

// Function to save activation vector to text file
void save_activations_to_txt(const std::string& filename, double* activations, int size) {
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(10);
    
    for (int i = 0; i < size; i++) {
        file << activations[i] << "\n";
    }
    
    file.close();
}

// Function to save single output activation
void save_single_activation_to_txt(const std::string& filename, double activation) {
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(10);
    file << activation << "\n";
    file.close();
}

// Main function to save all gradients and activations
void save_all_gradients_and_activations(
    // Gradient arrays
    double *grdntW740, double *grdntW512, double *grdntW256, 
    double *grdntW128, double *grdntW64, double *grdntW16,
    double *grdntB740, double *grdntB512, double *grdntB256,
    double *grdntB128, double *grdntB64, double grdntB16,
    // Activation arrays
    double *InputActivation, double *Actvtn512, double *Actvtn256,
    double *Actvtn128, double *Actvtn64, double *Actvtn16,
    double output_value)
{
    // Save weight gradients
    std::cout << "Saving weight gradients..." << std::endl;
    save_weight_gradient_to_csv("TestComparison/gradient_weights740_cpp.csv", grdntW740, 512, 740);
    save_weight_gradient_to_csv("TestComparison/gradient_weights512_cpp.csv", grdntW512, 256, 512);
    save_weight_gradient_to_csv("TestComparison/gradient_weights256_cpp.csv", grdntW256, 128, 256);
    save_weight_gradient_to_csv("TestComparison/gradient_weights128_cpp.csv", grdntW128, 64, 128);
    save_weight_gradient_to_csv("TestComparison/gradient_weights64_cpp.csv", grdntW64, 16, 64);
    save_weight_gradient_to_csv("TestComparison/gradient_weights16_cpp.csv", grdntW16, 1, 16);
    
    // Save bias gradients
    std::cout << "Saving bias gradients..." << std::endl;
    save_bias_gradient_to_csv("TestComparison/gradient_biases740_cpp.csv", grdntB740, 512);
    save_bias_gradient_to_csv("TestComparison/gradient_biases512_cpp.csv", grdntB512, 256);
    save_bias_gradient_to_csv("TestComparison/gradient_biases256_cpp.csv", grdntB256, 128);
    save_bias_gradient_to_csv("TestComparison/gradient_biases128_cpp.csv", grdntB128, 64);
    save_bias_gradient_to_csv("TestComparison/gradient_biases64_cpp.csv", grdntB64, 16);
    save_single_bias_gradient_to_csv("TestComparison/gradient_biases16_cpp.csv", grdntB16);
    
    // Save activations
    std::cout << "Saving activations..." << std::endl;
    save_activations_to_txt("TestComparison/activations_input_cpp.txt", InputActivation, 740);
    save_activations_to_txt("TestComparison/activations_layer1_512_cpp.txt", Actvtn512, 512);
    save_activations_to_txt("TestComparison/activations_layer2_256_cpp.txt", Actvtn256, 256);
    save_activations_to_txt("TestComparison/activations_layer3_128_cpp.txt", Actvtn128, 128);
    save_activations_to_txt("TestComparison/activations_layer4_64_cpp.txt", Actvtn64, 64);
    save_activations_to_txt("TestComparison/activations_layer5_16_cpp.txt", Actvtn16, 16);
    save_single_activation_to_txt("TestComparison/activations_output_cpp.txt", output_value);
    
    std::cout << "All files saved!" << std::endl;
    std::cout << "\nGradient files:" << std::endl;
    std::cout << "  - gradient_weights740_cpp.csv through gradient_weights16_cpp.csv" << std::endl;
    std::cout << "  - gradient_biases740_cpp.csv through gradient_biases16_cpp.csv" << std::endl;
    std::cout << "\nActivation files:" << std::endl;
    std::cout << "  - activations_input_cpp.txt (740 values)" << std::endl;
    std::cout << "  - activations_layer1_512_cpp.txt (512 values)" << std::endl;
    std::cout << "  - activations_layer2_256_cpp.txt (256 values)" << std::endl;
    std::cout << "  - activations_layer3_128_cpp.txt (128 values)" << std::endl;
    std::cout << "  - activations_layer4_64_cpp.txt (64 values)" << std::endl;
    std::cout << "  - activations_layer5_16_cpp.txt (16 values)" << std::endl;
    std::cout << "  - activations_output_cpp.txt (1 value)" << std::endl;
}

double *loadCsvToMatrix(const std::string &filename)
{
    rapidcsv::Document doc(filename, rapidcsv::LabelParams(-1, -1));

    // 2. Get Dimensions
    size_t out_rows = doc.GetRowCount();
    // We check the size of the first row to determine columns.

    size_t out_cols = doc.GetColumnCount();

    // 3. Allocate huge block on the heap
    // Flattened 2D array: Index = (row * num_cols) + col
    double *matrix = new double[out_rows * out_cols];

    // 4. Fill the array
    for (size_t i = 0; i < out_rows; ++i)
    {
        // Fetch the row using the rapidcsv syntax
        std::vector<double> row_vec = doc.GetRow<double>(i);

        // Safety check: ensure row length matches expected columns
        size_t copy_len = std::min(out_cols, row_vec.size());

        // Copy vector data to our heap array at the correct offset
        // Offset = i * out_cols
        std::copy(row_vec.begin(), row_vec.begin() + copy_len, matrix + (i * out_cols));

        // Handle edge case where a row might be shorter than expected (fill with 0 or NaN if needed)
        // Here we just leave uninitialized or strictly assume valid CSV.
    }

    return matrix;
}

/**
 * @brief Writes a flat heap-allocated array back to a CSV file.
 * * @param filename Output file path.
 * @param matrix Pointer to the heap array.
 * @param rows Number of rows.
 * @param cols Number of columns.
 */
void WriteMatrixToCsv(const std::string &filename, double *matrix, size_t rows, size_t cols)
{
    std::ofstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Error: Could not open " << filename << " for writing." << std::endl;
        return;
    }

    // CRITICAL: Set max precision to avoid data loss (truncation)
    file << std::setprecision(17);

    for (size_t r = 0; r < rows; ++r)
    {
        for (size_t c = 0; c < cols; ++c)
        {
            // Write value
            file << matrix[r * cols + c];

            // Write comma if not the last column
            if (c < cols - 1)
            {
                file << ",";
            }
        }
        // Newline at the end of the row
        file << "\n";
    }

    file.close();
}
