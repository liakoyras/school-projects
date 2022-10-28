#include <iostream>
//#include <cstdlib>
//#include <bit>

unsigned short count_ones(unsigned int a) {
    unsigned short count = 0;
    while (a > 0) {
        if ((a & 1) == 1)
            count++;
        a >>= 1;
    }
    return count;
}

int main() {
    //srand(42);

    std::cout << (count_ones(0b00000000000000000000000000000000) == 0) << std::endl;
    std::cout << (count_ones(0b00000000000000000000000000000001) == 1) << std::endl;
    std::cout << (count_ones(0b10000000000000000000000000000001) == 2) << std::endl;
    std::cout << (count_ones(0b10101010101010101010101010101010) == 16) << std::endl;
    std::cout << (count_ones(0b11111111111111111111111111111111) == 32) << std::endl;
    std::cout << (count_ones(0b10010011110100100110000100111101) == 16) << std::endl;

    return 0;
}
