#include <iostream>
#include <bit>
#include <ctime>


static const unsigned short n_tests=100;

// Calculate the number of set bits in an integer
unsigned short count_ones(unsigned int a) {
    unsigned short count = 0; // until only zero bits exist
    while (a > 0) {
        if ((a & 1) == 1) // mask to check the last bit
            count++;
        a >>= 1; //shift to the right
    }
    return count;
}

// Helper function to print the tests and results
void print_test(unsigned int input, unsigned short calculated, unsigned short expected) {
    std::cout << (expected == calculated ? "Pass" : "Fail") << " - ";
    std::cout << "Input: " << input << " ";
    std::cout << "Result: " << calculated << " ";
    std::cout << "Expected: " << expected << std::endl;
}

// Program driver
int main() {
    std::srand(std::time(NULL)); // different random seed each time
    
    // Testing uses std::popcount as the ground truth to calculate if
    // the values calculated by count_ones are correct
    
    // Set tests    
    print_test(0, count_ones(0), std::popcount<unsigned int>(0));
    print_test(1, count_ones(1), std::popcount<unsigned int>(1));
    print_test(2147483648, count_ones(2147483648), std::popcount<unsigned int>(2147483648));
    print_test(1599361067, count_ones(1599361067), std::popcount<unsigned int>(1599361067));
    print_test(2863311530, count_ones(2863311530), std::popcount<unsigned int>(2863311530));
    print_test(4294967295, count_ones(4294967295), std::popcount<unsigned int>(4294967295));
    
    std::cout << std::endl << "Random Tests" << std::endl;   

    // Random tests
    unsigned int test_number;
    for(int test = 0; test < n_tests; test++) {
        test_number = rand();
        print_test(test_number, count_ones(test_number), std::popcount<unsigned int>(test_number)); 
    }

    return 0;
}
