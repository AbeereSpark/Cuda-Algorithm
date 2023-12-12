#include <iostream>
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

// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define TPI 32
#define BITS 1024
#define INSTANCES 100

void printCgbnMem(const cgbn_mem_t<BITS>& value) {
    std::cout << "0x";
    for (int i = BITS / 32 - 1; i >= 0; --i) {
        std::cout << std::hex << std::setw(8) << std::setfill('0') << value._limbs[i];
    }
    std::cout << std::dec << std::endl;
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

struct KeyPair {
    cgbn_mem_t<BITS> private_key;
    cgbn_mem_t<BITS> public_key;
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

    // Read the public key as a string and convert to cgbn_mem_t
    cgbn_mem_t<BITS> publicKey;
    set_words(publicKey._limbs, argv[1], BITS / 32);

    // Read the operand as a string and convert to cgbn_mem_t
    cgbn_mem_t<BITS> operand;
    set_words(operand._limbs, argv[2], BITS / 32);

    char operationType = argv[3][0];
    int numIterations = std::stoi(argv[4]);
    const std::string matchFile = argv[5];

    std::cout << "Entered public key: " << argv[1] << std::endl;
    std::cout << "Entered operand: " << argv[2] << std::endl;

    // Read key pairs from bot.txt
    std::vector<KeyPair> botKeyPairs = readKeyPairs("bot.txt");

    bool matchFound = false;

    for (int iteration = 1; iteration <= numIterations && !matchFound; ++iteration) {
        std::cout << "Iteration Count: " << iteration << std::endl;

        // Perform the specified operation
        performOperation(publicKey, operand, operationType);

        // Display the result
        std::cout << "Result: ";
        printCgbnMem(publicKey);

        // Check if the result matches any public keys in bot.txt
        // for (const KeyPair& botKeyPair : botKeyPairs) {
        //     if (publicKey == botKeyPair.public_key) {
        //         // Match found, save the information to matchFile
        //         saveMatchToFile(matchFile, iteration, botKeyPair.public_key.get_str(16), publicKey.get_str(16));
        //         std::cout << std::endl << "Match found at Iteration " << iteration << std::endl;

        //         // Set the flag to true to exit both loops
        //         matchFound = true;
        //         break;
        //     }
        // }
    }


    return 0;
}
