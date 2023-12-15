// CPU Implementation CGBN
#include <iostream>
#include <iomanip>
#include <fstream> 
#include <sstream>
#include <vector>
#include <gmpxx.h>
#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"
#include "../utility/cpu_support.h"
#include "../utility/cpu_simple_bn_math.h"
#include "../utility/gpu_support.h"
#include <cuda_runtime.h>

// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define TPI 32
#define BITS (33 * 8)

// helpful typedefs for the kernel
typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

struct KeyPair {
    cgbn_mem_t<BITS> private_key;
    cgbn_mem_t<BITS> public_key;
};

// Function to convert cgbn_mem_t limbs to a hexadecimal string
std::string cgbnMemToString(const cgbn_mem_t<BITS>& value) {
    std::stringstream ss;
    ss << "0x";
    for (int i = BITS / 32 - 1; i >= 0; --i) {
        ss << std::hex << std::setw(8) << std::setfill('0') << value._limbs[i];
    }
    return ss.str();
}

// Helper function to perform addition or subtraction
void performOperation(cgbn_mem_t<BITS>& publicKey, cgbn_mem_t<BITS>& operand, char operation) {

    if (operation == 'A') {
        add_words(publicKey._limbs, publicKey._limbs, operand._limbs, BITS/32);
        // publicKey += operand;
    } else if (operation == 'S') {
        // publicKey -= operand;
        sub_words(publicKey._limbs, publicKey._limbs, operand._limbs, BITS/32);
    }
}

// Helper function to read key pairs from a file
std::vector<KeyPair> readKeyPairs(const std::string& filename) {
    std::vector<KeyPair> keyPairs;
    std::ifstream file(filename);

    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string token;
            KeyPair keyPair;

            // Read Private_k label
            iss >> token;  // Read the label
            if (token != "Private_k:") {
                // Handle error or skip line
                continue;
            }

            // Read private key (in hexadecimal format)
            set_words(keyPair.private_key._limbs, "", BITS / 32);
            iss >> token;
            set_words(keyPair.private_key._limbs, token.c_str(), BITS / 32);

            // Read Public_k label
            iss >> token;  // Read the label
            if (token != "Public_k:") {
                // Handle error or skip line
                continue;
            }

            // Read public key (in hexadecimal format)
            set_words(keyPair.public_key._limbs, "", BITS / 32);
            iss >> token;
            set_words(keyPair.public_key._limbs, token.c_str(), BITS / 32);

            keyPairs.push_back(keyPair);
        }
        file.close();
    }

    return keyPairs;
}


// Helper function to save the matched public key, iteration count, and result to a file
void saveMatchToFile(const std::string& matchFile, const std::string& iteration, const std::string& publicKey) {
    std::ofstream file(matchFile, std::ios::app);
    if (file.is_open()) {
        file << "Iteration Count: " << iteration << std::endl;
        file << "Matched Public Key: [" << publicKey << "]" << std::endl;
        file.close();
    }
}

// GPU kernel for comparing results in parallel
__global__ void kernel_compare(cgbn_error_report_t *report, cgbn_mem_t<BITS>* results, KeyPair* botKeyPairs, int numResults, bool* matchFound) {
    int instance = (blockIdx.x * blockDim.x + threadIdx.x )/ TPI;

    if ((instance < numResults) && !(*matchFound)) 
    {
        cgbn_mem_t<BITS> publicKey = results[0];
        cgbn_mem_t<BITS>& botPublicKey = botKeyPairs[instance].public_key;

        context_t      bn_context(cgbn_report_monitor, report, instance);   // construct a context
        env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
        env_t::cgbn_t  a, b;                                             // define a, b, r as 1024-bit bignums

        cgbn_load(bn_env, a, &publicKey);      // load my instance's a value
        cgbn_load(bn_env, b, &botPublicKey);      // load my instance's b value

        int comparisonResult = cgbn_equals(bn_env, a, b);

        if (comparisonResult) {
            *matchFound = true;
        }
    }
}

// Function to perform GPU comparison
bool performGPUComparison(cgbn_mem_t<BITS>* h_results, const std::vector<KeyPair>& botKeyPairs) {
    bool matchFound = false;  // Variable to control the loop

    cgbn_mem_t<BITS>* d_results;
    KeyPair* d_botKeyPairs;
    bool* d_matchFound;
    cgbn_error_report_t *report;

    // Allocate memory on the GPU
    CUDA_CHECK(cudaMalloc((void**)&d_results, 1 * sizeof(cgbn_mem_t<BITS>)));
    CUDA_CHECK(cudaMalloc((void**)&d_botKeyPairs, botKeyPairs.size() * sizeof(KeyPair)));
    CUDA_CHECK(cudaMalloc((void**)&d_matchFound, sizeof(bool)));

    // Copy data to the GPU
    CUDA_CHECK(cudaMemcpy(d_results, h_results, 1 * sizeof(cgbn_mem_t<BITS>), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_botKeyPairs, botKeyPairs.data(), botKeyPairs.size() * sizeof(KeyPair), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_matchFound, &matchFound, sizeof(bool), cudaMemcpyHostToDevice));

    // create a cgbn_error_report for CGBN to report back errors
    CUDA_CHECK(cgbn_error_report_alloc(&report)); 

    int numResults = botKeyPairs.size();

    // Launch the GPU kernel
    int block_size = 4;
    int num_blocks = (numResults + block_size - 1) / block_size;
    kernel_compare<<<num_blocks, block_size * TPI>>>(report, d_results, d_botKeyPairs, numResults, d_matchFound);

    // Wait for the kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back the result
    CUDA_CHECK(cudaMemcpy(&matchFound, d_matchFound, sizeof(bool), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_botKeyPairs));
    CUDA_CHECK(cudaFree(d_matchFound));

    return matchFound;
}

bool checkCudaAvailability() {
    int deviceCount;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);

    if (cudaStatus != cudaSuccess) {
        std::cerr << "Error getting device count: " << cudaGetErrorString(cudaStatus) << std::endl;
        return false;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA-enabled GPU device found." << std::endl;
        return false;
    }

    std::cout << "Found " << deviceCount << " CUDA-enabled GPU device(s)." << std::endl;

    // You can also print more information about each device if needed
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        // Print more properties if needed
    }

    return true;
}

__global__ void kernel_iterate(cgbn_error_report_t *report, cgbn_mem_t<BITS>* publicKeys, const cgbn_mem_t<BITS>* operands, int numIterations, KeyPair* botKeyPairs, int numResults,bool* matchFound) {
    uint32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) ;
    cgbn_mem_t<BITS> iterationValue;
    iterationValue._limbs[0] = instance;
    cgbn_mem_t<BITS> alteredKey;

    if ((instance < numResults)) {
        cgbn_mem_t<BITS> publicKey = publicKeys[0];
        cgbn_mem_t<BITS> operand = operands[0];

        context_t      bn_context(cgbn_report_monitor, report, instance);   // construct a context
        env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
        env_t::cgbn_t  pKey, op, rAdd, iter;                                             // define a, b, r as 1024-bit bignums
        env_t::cgbn_wide_t rMul;

        cgbn_load(bn_env, pKey, &publicKey);      // load my instance's a value
        cgbn_load(bn_env, op, &operand);      // load my instance's b value
        cgbn_load(bn_env, iter, &iterationValue);      // load my instance's b value

        // Generate a new key by adding (operand * iteration) to the public key

        cgbn_mul_wide(bn_env, rMul, iter, op);

        cgbn_add(bn_env, rAdd, pKey, rMul._low);

        cgbn_store(bn_env, &alteredKey, rAdd);   

        // Now, launch the compare kernel to check for matches
        // Launch the GPU kernel
        int block_size = 4;
        int num_blocks = (numResults + block_size - 1) / block_size;
        kernel_compare(report, &rAdd, botKeyPairs, numResults, matchFound);
    }
}

int main(int argc, char* argv[]) {
    
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <public_key> <operation_value> <operation_type(A/S)> <num_iterations> <match_file>\n";
        return 1;
    }

    if (checkCudaAvailability()) {
        // Perform GPU-related tasks here
        std::cout << "GPU is available. Proceed with GPU-related tasks." << std::endl;
    } else {
        // Perform CPU-only tasks here
        std::cout << "No GPU available. Proceed with CPU-only tasks." << std::endl;
    }

    // Read the public key as a string and convert to cgbn_mem_t
    cgbn_mem_t<BITS> publicKey;
    set_words(publicKey._limbs, argv[1], BITS / 32);

    // Read the operand as a string and convert to cgbn_mem_t
    cgbn_mem_t<BITS> operand;
    set_words(operand._limbs, argv[2], BITS / 32);

    char operationType = argv[3][0];
    
    cgbn_mem_t<BITS> numIterations;
    set_words(numIterations._limbs, argv[4], BITS / 32);

    const std::string matchFile = argv[5];

    std::cout << "Entered public key: " << argv[1] << std::endl;
    std::cout << "Entered operand: " << argv[2] << std::endl << std::endl;

    // Read key pairs from bot.txt
    std::vector<KeyPair> botKeyPairs = readKeyPairs("bot.txt");

    bool matchFound = false;

    performOperation(publicKey, operand, operationType);

    // Check if the result matches any public keys in bot.txt
    matchFound = performGPUComparison(&publicKey, botKeyPairs);

    if (matchFound) {
        // std::cout << std::endl << "Match found at Iteration " << cgbnMemToString(iteration) << std::endl;
        // saveMatchToFile(matchFile, cgbnMemToString(iteration), cgbnMemToString(publicKey));
    }

    else {
        std::cout << std::endl << "No Match found " << std::endl;
    }


    return 0;
}

