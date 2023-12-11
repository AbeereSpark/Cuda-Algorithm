#include <iostream>
#include <gmpxx.h>

// Helper function to perform addition or subtraction
void performOperation(mpz_class& publicKey, const mpz_class& operand, char operation) {
    if (operation == 'A') {
        publicKey += operand;
    } else if (operation == 'S') {
        publicKey -= operand;
    }
}

// Helper function to read public keys from a file
std::vector<std::string> readPublicKeys(const std::string& filename) {
    std::vector<std::string> publicKeys;
    std::ifstream file(filename);

    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            publicKeys.push_back(line);
        }
        file.close();
    }

    return publicKeys;
}

// Helper function to save the matched public key, iteration count, and result to a file
void saveMatchToFile(const std::string& matchFile, int iteration, const std::string& publicKey, const std::string& result) {
    std::ofstream file(matchFile, std::ios::app);
    if (file.is_open()) {
        file << "Iteration Count: " << iteration << std::endl;
        file << "Matched Public Key: [" << publicKey << "]" << std::endl;
        file << "Result: " << result << std::endl << std::endl;
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

    std::cout << "Enter the public key: " << publicKey.get_str(16) << std::endl;
    std::cout << "Choose between addition or subtraction: " << std::endl;

    // Read public keys from bot.txt
    std::vector<std::string> botPublicKeys = readPublicKeys("bot.txt");

    for (int iteration = 1; iteration <= numIterations; ++iteration) {
        std::cout << "Iteration Count: " << iteration << std::endl;

        // Perform the specified operation
        performOperation(publicKey, operand, operationType);

        // Display the result
        std::cout << "Result: " << publicKey.get_str(16) << std::endl;

        // Check if the result matches any public keys in bot.txt
        for (const std::string& botPublicKey : botPublicKeys) {
            if (publicKey.get_str(16) == botPublicKey) 
            {
                // Match found, save the information to matchFile
                saveMatchToFile(matchFile, iteration, botPublicKey, publicKey.get_str(16));
                std::cout << "Match found at Iteration " << iteration << std::endl;
                // Optionally, you may choose to exit the loop or program here
            }
        }
    }

    return 0;
}
