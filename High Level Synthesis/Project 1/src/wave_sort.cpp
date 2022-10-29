#include <iostream>

// Swap two numbers in memory. 
void swap(int &a, int &b) {
    int t = a;
    a = b;
    b = t;
}

// Sort an array of integers
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

// Sort an array in wave form, so that odd-indexed elements are  less
// than (or equal to) the previous even-indexed element and less than
// (or equal to) the next element.
template<int N>
void wave_sort(int a[N]) {
    sort<N>(a); // sort the array
    for(int i = 0; i < N -  1; i+=2){
        swap(a[i], a[i+1]); // swap every two elements for the condition
    }
}


// Helper function to print an 1D array.
template<int N>
void print_array(int a[N]) {
    for(int i = 0; i < N; i++) {
        std::cout << a[i] << "  ";
    }
    std::cout << std::endl;
}

// Helper function that compares the values of two arrays.
// Returns true if they are the same, false otherwise.
template<int N>
bool compare_arrays(int a[N], int b[N]) {
    for(int i = 0; i < N; i++) {
        if(a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

// Helper function to print the tests and results.
template<int N>
void print_test(int input[N], int calculated[N], int expected[N]) {
    std::cout << (compare_arrays<N>(calculated, expected) ? "Pass" : "Fail") << std::endl;
    std::cout << "Input:    ";
    print_array<N>(input);
    std::cout << "Result:   ";
    print_array<N>(calculated);
    std::cout << "Expected: ";
    print_array<N>(expected);
}

// Program driver
int main() {
    // Tests
    int a[5] = {5, 2, 9, 3, 2};
    int b[5] = {5, 2, 9, 3, 2};
    int c[5] = {2, 2, 5, 3, 9};
    wave_sort<5>(b);
    print_test<5>(a, b, c);

    std::cout << std::endl;

    int d[6] = {3, 2, 9, 6, 4, 1};
    int e[6] = {3, 2, 9, 6, 4, 1};
    int f[6] = {2, 1, 4, 3, 9, 6};
    wave_sort<6>(e);
    print_test<6>(d, e, f);

    std::cout << std::endl;
    
    int g[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int h[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int i[10] = {2, 1, 4, 3, 6, 5, 8, 7, 10, 9};
    wave_sort<10>(h);
    print_test<10>(g, h, i);

    std::cout << std::endl;

    int j[10] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    int k[10] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    int l[10] = {2, 1, 4, 3, 6, 5, 8, 7, 10, 9};
    wave_sort<10>(k);
    print_test<10>(j, k, l);

    std::cout << std::endl;

    int m[10] = {1, 5, 7, 2, 3, 8, 10, 4, 9, 6};
    int n[10] = {1, 5, 7, 2, 3, 8, 10, 4, 9, 6};
    int o[10] = {2, 1, 4, 3, 6, 5, 8, 7, 10, 9};
    wave_sort<10>(n);
    print_test<10>(m, n, o);

    std::cout << std::endl;
    
    int p[20] = {-5116, 1123, -3646, 7522, -6566, 8985, 1528, -3546, 5714, 7769, 5236, -5119, 1026, -4072, 8194, -2851, 2203, -9540, -4139, 4041};
    int q[20] = {-5116, 1123, -3646, 7522, -6566, 8985, 1528, -3546, 5714, 7769, 5236, -5119, 1026, -4072, 8194, -2851, 2203, -9540, -4139, 4041};
    int r[20] = {-6566, -9540, -5116, -5119, -4072, -4139, -3546, -3646, 1026, -2851, 1528, 1123, 4041, 2203, 5714, 5236, 7769, 7522, 8985, 8194};
    wave_sort<20>(q);
    print_test<10>(p, q, r);
    
    return 0;
}
