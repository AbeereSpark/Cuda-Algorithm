#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <gmpxx.h>

// CUDA kernel for performing addition or subtraction on the GPU
__global__ void performOperationKernel(mpz_class* publicKey, const mpz_class operand, char operation, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        if (operation == 'A') {
            publicKey[tid] += operand;
        } else if (operation == 'S') {
            publicKey[tid] -= operand;
        }
    }
}

// Helper function to perform addition or subtraction on the GPU
void performOperationGPU(mpz_class* publicKey, const mpz_class operand, char operation, int size) {
    mpz_class* d_publicKey;
    cudaMalloc((void**)&d_publicKey, size * sizeof(mpz_class));
    cudaMemcpy(d_publicKey, publicKey, size * sizeof(mpz_class), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    performOperationKernel<<<gridSize, blockSize>>>(d_publicKey, operand, operation, size);

    cudaMemcpy(publicKey, d_publicKey, size * sizeof(mpz_class), cudaMemcpyDeviceToHost);
    cudaFree(d_publicKey);
}

struct KeyPair {
    mpz_class private_key;
    mpz_class public_key;
};

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
            iss >> std::hex >> keyPair.private_key;

            // Read Public_k label
            iss >> token;  // Read the label
            if (token != "Public_k:") {
                // Handle error or skip line
                continue;
            }

            // Read public key (in hexadecimal format)
            iss >> std::hex >> keyPair.public_key;

            keyPairs.push_back(keyPair);
        }
        file.close();
    }

    return keyPairs;
}

// Helper function to save the matched public key, iteration count, and result to a file
void saveMatchToFile(const std::string& matchFile, int iteration, const std::string& publicKey, const std::string& result) {
    std::ofstream file(matchFile, std::ios::app);
    if (file.is_open()) {
        file << "Iteration Count: " << iteration << std::endl;
        file << "Matched Public Key: [" << publicKey << "]" << std::endl;
        file.close();
    }
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <public_key> <operation_value> <operation_type(A/S)> <num_iterations> <match_file>\n";
        return 1;
    }

    mpz_class publicKey(argv[1], 16);
    mpz_class operand(argv[2]);
    char operationType = argv[3][0];
    int numIterations = std::stoi(argv[4]);
    const std::string matchFile = argv[5];

    std::cout << "Entered public key: " << publicKey.get_str(16) << std::endl;

    // Read key pairs from bot.txt
    std::vector<KeyPair> botKeyPairs = readKeyPairs("bot.txt");

    bool matchFound = false;

    for (int iteration = 1; iteration <= numIterations && !matchFound; ++iteration) {
        std::cout << "Iteration Count: " << iteration << std::endl;

        // Perform the specified operation on GPU
        performOperationGPU(&publicKey, operand, operationType, 1);

        // Display the result
        std::cout << "Result: " << publicKey.get_str(16) << std::endl;

        // Check if the result matches any public keys in bot.txt
        for (const KeyPair& botKeyPair : botKeyPairs) {
            if (publicKey == botKeyPair.public_key) {
                // Match found, save the information to matchFile
                saveMatchToFile(matchFile, iteration, botKeyPair.public_key.get_str(16), publicKey.get_str(16));
                std::cout << std::endl << "Match found at Iteration " << iteration << std::endl;

                // Set the flag to true to exit both loops
                matchFound = true;
                break;
            }
        }
    }

    return 0;
}
