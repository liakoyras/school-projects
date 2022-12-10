#include <iostream>
#include "ac_int.h"

#include "mc_scverify.h"

template <int N>
ac_int<8, false> CCS_BLOCK(binary_search)(ac_int<8, false> a[N], ac_int<8, false> target) {
    int left = 0;
    int right = N-1;
    int mid;

    for(int i = 0; i < N/2 + 1; i++) {
        mid = (right + left) / 2;
        if((a[mid] == target) or (right <= left)) {
            return mid;
        } else {
            if(a[mid] > target) {
                right = mid;
            } else {
                left = mid;
            }
        }
    }
    return N;
}

CCS_MAIN(int argc, char* argv[]) {
    ac_int<8, false> a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    ac_int<8, false> result = binary_search<10>(a, 2);

    std::cout << "2 is in position " << result << std::endl;

    result = binary_search<10>(a, 9);
    
    std::cout << "9 is in position " << result << std::endl;
    
    result = binary_search<10>(a, 20);
    
    std::cout << "20 is in position " << result << std::endl;
    
    CCS_RETURN(0);
}
