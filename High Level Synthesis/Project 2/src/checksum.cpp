#include <iostream>

// Calculate the checksum of an integer. Digits in odd-numbered positions
// are doubled and if this results to a number exceeding 9, both resulting
// digits are summed and added to the checksum.
unsigned int checksum(unsigned int a) {
    unsigned int sum = 0;
    unsigned short digit_position = 0;
    unsigned short digit_sum = 0;

    do {
        if(digit_position % 2 == 0) { // check even-number positions
            digit_sum = (a % 10); // mod 10 takes the final digit
        } else {
            digit_sum = 2 * (a % 10); // double the digit
            if(digit_sum >= 10) { // if the product is >= 10
                digit_sum = (digit_sum % 10) + (digit_sum / 10); // sum both digits
            }
        }
        sum += digit_sum;
        digit_position++;
    }while(a /= 10); // shift to the right

    return sum;
}


// Helper function to print the tests
void print_test(unsigned int input, unsigned int calculated, unsigned int expected) {
    std::cout << (expected == calculated ? "Pass" : "Fail") << " - ";
    std::cout << "Input: " << input << " ";
    std::cout << "Result: " << calculated << " ";
    std::cout << "Expected: " << expected << std::endl;
}

// Program driver
int main() {
    // Tests
    print_test(3561, checksum(3561), 15);
    print_test(0, checksum(0), 0);
    print_test(9, checksum(9), 9);
    print_test(21, checksum(21), 5);
    print_test(35, checksum(35), 11);
    print_test(55, checksum(55), 6);
    print_test(65, checksum(65), 8);
    print_test(4623165, checksum(4623165), 24);
    print_test(1654684, checksum(1654684), 34);
    print_test(11111, checksum(11111), 7);
    print_test(2300, checksum(2300), 7);
    print_test(55555, checksum(55555), 17);
    print_test(69821103, checksum(69821103),27);
    print_test(35840517, checksum(35840517), 36);
    print_test(999999999, checksum(999999999), 81);

    return 0;
}
