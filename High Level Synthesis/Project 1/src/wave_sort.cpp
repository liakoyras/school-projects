#include <iostream>
//#include <cstdlib>
//#include <bit>

void swap(int &a, int &b) {
    int t = a;
    a = b;
    b = t;
}

template<int N>
void sort(int a[N]) {
    // bubble sort
    for(int i = 0; i < N - 1; i++) {
        for(int j = 0; j < N - i - 1; j++) {
            if(a[j] > a[j+1]) {
                swap(a[j], a[j+1]);
            }
        }
    }
}


template<int N>
void wave_sort(int a[N]) {
    sort<N>(a);
    for(int i = 0; i < N -  1; i+=2){
        swap(a[i], a[i+1]);
    }
}

template<int N>
void print_array(int a[N]) {
    for (int i = 0; i < N; i++) {
        std::cout << a[i] << " "; 
    }
    std::cout << std::endl;
} 

int main() {
    int a[5] = {5, 2, 9, 3, 2};
    std::cout << "Original: ";
    print_array<5>(a);
    wave_sort<5>(a);
    std::cout << "Wave: ";
    print_array<5>(a);

    std::cout << std::endl;

    int b[6] = {3, 2, 9, 6, 4, 1};
    std::cout << "Original: ";
    print_array<6>(b);
    wave_sort<6>(b);
    std::cout << "Wave: ";
    print_array<6>(b);

    std::cout << std::endl;
    
    int c[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::cout << "Original: ";
    print_array<10>(c);
    wave_sort<10>(c);
    std::cout << "Wave: ";
    print_array<10>(c);

    std::cout << std::endl;

    int d[10] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    std::cout << "Original: ";
    print_array<10>(d);
    wave_sort<10>(d);
    std::cout << "Wave: ";
    print_array<10>(d);

    std::cout << std::endl;

    int e[10] = {1, 5, 7, 2, 3, 8, 10, 4, 9, 6};
    std::cout << "Original: ";
    print_array<10>(e);
    wave_sort<10>(e);
    std::cout << "Wave: ";
    print_array<10>(e);

    return 0;
}
