#include <iostream>

// Compute the sum of each row of a 2D array and save it to another.
template<int N, int M>
void compute_row_sum(short a[N][M], short row_sum[N]) {
    int sum;
    for(int i = 0; i < N; i++) {
        sum = 0;
        for(int j = 0; j < M; j++) {
            sum += a[i][j];
        }
        row_sum[i] = sum;
    }
}


// Helper function to print a 2D array
template<int N, int M>
void print_array_2d(short a[N][M]) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            std::cout << a[i][j] << "  ";
        }
        std::cout << std::endl;
    }
}

// Helper function to print an 1D array
template<int N>
void print_array(short a[N]) {
    for(int i = 0; i < N; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
}

// Helper function to compare the values of two arrays.
// Returns true if they are the same, false otherwise.
template<int N>
bool compare_arrays(short a[N], short b[N]) {
    for(int i = 0; i < N; i++) {
        if(a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

// Helper function to print the tests and results
template<int N, int M>
void print_test(short input[N][M], short calculated[N], short expected[N]) {
    std::cout << (compare_arrays<N>(calculated, expected) ? "Pass" : "Fail") << std::endl;
    std::cout << "Input: " << std::endl;
    print_array_2d<N, M>(input);
    std::cout << "Result:   ";
    print_array<N>(calculated);
    std::cout << "Expected: ";
    print_array<N>(expected);
}

// Program driver
int main() {
    // Tests
    short a[1][3] = {{1, 2, 3}};
    short b[1];
    short c[1] = {6};
    compute_row_sum<1,3>(a, b);
    print_test<1,3>(a, b, c);

    std::cout << std::endl; 

    short d[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    short e[3];
    short f[3] = {6, 15, 24};
    compute_row_sum<3,3>(d, e);
    print_test<3,3>(d, e, f);
    
    std::cout << std::endl;

    short g[2][2] = {{1, -2}, {-3, 4}};
    short h[2];
    short i[2] = {-1, 1};
    compute_row_sum<2,2>(g, h);
    print_test<2,2>(g, h, i);
    
    std::cout << std::endl;

    short j[3][1] = {{1}, {2}, {3}};
    short k[3];
    short l[3] = {1, 2, 3};
    compute_row_sum<3,1>(j, k);
    print_test<3,1>(j, k, l);

    std::cout << std::endl;

    short m[5][2] = {{1561, 8864}, {5958, 3297}, {7680, 9000}, {2291, 5258},{67, 10617}};  
    short n[5];
    short o[5] = {10425, 9255, 16680, 7549, 10684};
    compute_row_sum<5,2>(m, n);
    print_test<5, 2>(m, n, o);

    return 0;
}
