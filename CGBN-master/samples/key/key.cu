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
#include <chrono>

// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define TPI 1
#define BITS (36 * 8)

// helpful typedefs for the kernel
typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

struct KeyPair {
    cgbn_mem_t<BITS> private_key;
    cgbn_mem_t<BITS> public_key;
};

void printMaxLimits() {
    int maxBlocks, maxThreadsPerBlock;

    // Get the maximum number of blocks per grid
    cudaDeviceGetAttribute(&maxBlocks, cudaDevAttrMaxGridDimX, 0);
    std::cout << "Maximum blocks per grid: " << maxBlocks << std::endl;

    // Get the maximum number of threads per block
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    std::cout << "Maximum threads per block: " << maxThreadsPerBlock << std::endl;
}

// // Function to convert cgbn_mem_t limbs to a hexadecimal string
std::string cgbnMemToStringCPU(const cgbn_mem_t<BITS>& value) {
    std::stringstream ss;
    ss << "0x";
    for (int i = BITS / 32 - 1; i >= 0; --i) {
        ss << std::hex << std::setw(8) << std::setfill('0') << value._limbs[i];
    }
    return ss.str();
}

// Function to convert an integer to a hexadecimal string
__host__ __device__ void intToHexStr(uint32_t value, char* output) {
    const char hexChars[] = "0123456789abcdef";
    output[0] = hexChars[(value >> 28) & 0xF];
    output[1] = hexChars[(value >> 24) & 0xF];
    output[2] = hexChars[(value >> 20) & 0xF];
    output[3] = hexChars[(value >> 16) & 0xF];
    output[4] = hexChars[(value >> 12) & 0xF];
    output[5] = hexChars[(value >> 8) & 0xF];
    output[6] = hexChars[(value >> 4) & 0xF];
    output[7] = hexChars[value & 0xF];
    output[8] = '\0';
}

// Function to convert cgbn_mem_t limbs to a hexadecimal string
__host__ __device__ void cgbnMemToStringGPU(const cgbn_mem_t<BITS>& value, char* output) {
    int index = 0;
    for (int i = BITS / 32 - 1; i >= 0; --i) {
        intToHexStr(value._limbs[i], output + index);
        index += 8;
    }
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
__global__ void kernel_compare(cgbn_error_report_t *report, cgbn_mem_t<BITS> publicKey, KeyPair* botKeyPairs, cgbn_mem_t<BITS>* matchedKey, uint32_t numResults, bool* matchFound, uint32_t iterInstanceCount, int* iterCount) {
    int instance = (blockIdx.x * blockDim.x + threadIdx.x )/ TPI;

    if ((instance < numResults) && !(*matchFound)) 
    {
        cgbn_mem_t<BITS>& botPublicKey = botKeyPairs[instance].public_key;

        context_t      bn_context(cgbn_report_monitor, report, instance);   // construct a context
        env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
        env_t::cgbn_t  a, b;                                             // define a, b, r as 1024-bit bignums

        cgbn_load(bn_env, a, &publicKey);      // load my instance's a value
        cgbn_load(bn_env, b, &botPublicKey);      // load my instance's b value

        int comparisonResult = cgbn_equals(bn_env, a, b);

        if (comparisonResult) {
            *matchFound = true;
            *iterCount = iterInstanceCount;
            cgbn_store(bn_env, matchedKey, a);   // store r into sum
        }
    }
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

__global__ void kernel_iterate(cgbn_error_report_t *report, cgbn_mem_t<BITS>* publicKeys, KeyPair* botKeyPairs, cgbn_mem_t<BITS>* matchedKey, char operationType, const cgbn_mem_t<BITS>* operands, uint32_t numIterations, int numResults, bool* matchFound, int* iterCount) {
    uint32_t instance = (blockIdx.x * blockDim.x + threadIdx.x);
    // cgbn_mem_t<BITS> iterationValue;
    // iterationValue._limbs[0] = instance;
    cgbn_mem_t<BITS> alteredKey;

    if (((instance < numIterations) && !(*matchFound))) {
        cgbn_mem_t<BITS> publicKey = publicKeys[0];
        cgbn_mem_t<BITS> operand = operands[0];

        typedef cgbn_context_t<1>         context_single_t;
        typedef cgbn_env_t<context_single_t, BITS> env_single_t;
        context_single_t      bn_context(cgbn_report_monitor, report, instance);   // construct a context
        env_single_t          bn_env(bn_context.env<env_single_t>());                     // construct an environment for 1024-bit math
        env_single_t::cgbn_t  pKey, op, r, iter;                                             // define a, b, r as 1024-bit bignums
        // env_single_t::cgbn_t rMul;

        cgbn_load(bn_env, pKey, &publicKey);      // load my instance's a value
        cgbn_load(bn_env, op, &operand);      // load my instance's b value
        // cgbn_load(bn_env, iter, &iterationValue);      // load my instance's b value

        cgbn_mul_ui32(bn_env, iter, op, instance);
        // Generate a new key by adding (operand * iteration) to the public key

        // cgbn_mul(bn_env, r, iter, op);

        if (operationType == 'A') 
        {
            cgbn_add(bn_env, r, pKey, iter);
        } 
        else if (operationType == 'S') 
        {
            cgbn_sub(bn_env, r, pKey, iter);     
        }    

        cgbn_store(bn_env, &alteredKey, r);   

        // Now, launch the compare kernel to check for matches
        // Launch the GPU kernel
        uint32_t block_size = 512;
        uint32_t num_blocks = (numResults + block_size - 1) / block_size;
        // char pString[100];
        // cgbnMemToStringGPU(alteredKey, pString);
        // printf("0x%s\n", pString);
        kernel_compare<<<num_blocks, block_size * TPI>>>(report, alteredKey, botKeyPairs, matchedKey, numResults, matchFound, instance, iterCount);
    }
}

// Function to perform GPU comparison
bool performGPUComparison(cgbn_mem_t<BITS>* h_publicKey, const std::vector<KeyPair>& botKeyPairs, char operationType, cgbn_mem_t<BITS>* h_operand, uint32_t numIterations, const std::string matchFile) {
    bool matchFound = false;  // Variable to control the loop
    int iterCount = 0;  // Variable to control the loop
    cgbn_mem_t<BITS> matchedKey;

    cgbn_mem_t<BITS>* d_publicKey;
    cgbn_mem_t<BITS>* d_operand;
    cgbn_mem_t<BITS>* d_matchedKey;
    KeyPair* d_botKeyPairs;
    bool* d_matchFound;
    int* d_iterCount;
    cgbn_error_report_t *report;

    // Allocate memory on the GPU
    CUDA_CHECK(cudaMalloc((void**)&d_publicKey, sizeof(cgbn_mem_t<BITS>)));
    CUDA_CHECK(cudaMalloc((void**)&d_operand, sizeof(cgbn_mem_t<BITS>)));
    CUDA_CHECK(cudaMalloc((void**)&d_matchedKey, sizeof(cgbn_mem_t<BITS>)));
    CUDA_CHECK(cudaMalloc((void**)&d_botKeyPairs, botKeyPairs.size() * sizeof(KeyPair)));
    CUDA_CHECK(cudaMalloc((void**)&d_matchFound, sizeof(bool)));
    CUDA_CHECK(cudaMalloc((void**)&d_iterCount, sizeof(int)));

    // Copy data to the GPU
    CUDA_CHECK(cudaMemcpy(d_publicKey, h_publicKey, sizeof(cgbn_mem_t<BITS>), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_operand, h_operand, sizeof(cgbn_mem_t<BITS>), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_botKeyPairs, botKeyPairs.data(), botKeyPairs.size() * sizeof(KeyPair), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_matchFound, &matchFound, sizeof(bool), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_iterCount, &iterCount, sizeof(int), cudaMemcpyHostToDevice));

    // create a cgbn_error_report for CGBN to report back errors
    CUDA_CHECK(cgbn_error_report_alloc(&report)); 

    uint32_t numResults = botKeyPairs.size();

    // Before the kernel launch
    cudaError_t cudaStatus;

    // Launch the GPU kernel
    uint32_t block_size = 512;
    uint32_t num_blocks = (numIterations + block_size - 1U) / block_size;
    kernel_iterate<<<num_blocks, block_size>>>(report, d_publicKey, d_botKeyPairs, d_matchedKey, operationType, d_operand, numIterations, numResults, d_matchFound, d_iterCount);

    // Check for kernel launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Additional error handling or debugging steps can be added here
    }
    // Wait for the kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back the result
    CUDA_CHECK(cudaMemcpy(&matchFound, d_matchFound, sizeof(bool), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&iterCount, d_iterCount, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&matchedKey, d_matchedKey, sizeof(cgbn_mem_t<BITS>), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_publicKey));
    CUDA_CHECK(cudaFree(d_botKeyPairs));
    CUDA_CHECK(cudaFree(d_matchFound));
    CUDA_CHECK(cudaFree(d_iterCount));

    if (matchFound) {
        std::cout << std::endl << "Match found at Iteration " << iterCount << std::endl;
        saveMatchToFile(matchFile, std::to_string(iterCount), cgbnMemToStringCPU(matchedKey));
    }

    return matchFound;
}

int main(int argc, char* argv[]) {
    
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <public_key> <operation_value> <operation_type(A/S)> <num_iterations> <match_file>\n";
        return 1;
    }

    if (checkCudaAvailability()) {
        // Perform GPU-related tasks here
        std::cout << "GPU is available. Proceed with GPU-related tasks." << std::endl;
        printMaxLimits();
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
    uint64_t numIterationsInt = 0;
    memcpy(&numIterationsInt, numIterations._limbs, sizeof(uint64_t));

    const std::string matchFile = argv[5];

    std::cout << "Entered public key: " << argv[1] << std::endl;
    std::cout << "Entered operand: " << argv[2] << std::endl << std::endl;
    std::cout << "Entered Number of Iterations: " << cgbnMemToStringCPU(numIterations) << std::endl << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();  // Record the start time

    // Read key pairs from bot.txt
    std::vector<KeyPair> botKeyPairs = readKeyPairs("bot.txt");

    // performOperation(publicKey, operand, operationType);

    // Check if the result matches any public keys in bot.txt
    bool matchResult = performGPUComparison(&publicKey, botKeyPairs, operationType, &operand, numIterations._limbs[0], matchFile);

    if (!matchResult){
        std::cout << std::endl << "No Match found " << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();  // Record the end time
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);  // Calculate the duration in milliseconds
    std::cout << std::endl << "Program duration: " << duration.count() << " milliseconds" << std::endl;
    std::cout << std::endl << "Program duration: " << duration.count() / 1000.0 << " seconds" << std::endl;



    return 0;
}
