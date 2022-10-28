#include <iostream>

unsigned int checksum(unsigned int a) {
    unsigned int sum = 0;
    unsigned short digit = 0;
    unsigned short digit_sum = 0;

    do {
        digit++;

        if(digit % 2 != 0) {
            digit_sum = (a % 10);
        } else {
            digit_sum = 2 * (a % 10);
            if(digit_sum >= 10) {
                digit_sum = (digit_sum % 10) + (digit_sum / 10);
            }
        }
        sum += digit_sum;
    }while(a /= 10);

    return sum;
}


int main() {
    std::cout << (checksum(3561) == 15) << std::endl;
    std::cout << (checksum(0) == 0) << std::endl;
    std::cout << (checksum(9) == 9) << std::endl;
    std::cout << (checksum(21) == 5) << std::endl;
    std::cout << (checksum(35) == 11) << std::endl;
    std::cout << (checksum(55) == 6) << std::endl;
    std::cout << (checksum(65) == 8) << std::endl;
    //std::cout << (checksum(3561) == 15) << std::endl;
    //std::cout << (checksum(3561) == 15) << std::endl;
    return 0;
}
