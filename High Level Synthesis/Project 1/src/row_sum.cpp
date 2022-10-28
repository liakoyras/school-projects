#include <iostream>

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


int main() {
    short a[1][3] = {{1, 2, 3}};
    short b[1];
    compute_row_sum<1,3>(a, b);
    std::cout << (b[0] == 6) << std::endl;

    short c[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    short d[1];
    compute_row_sum<3,3>(c, d);
    std::cout << (d[0] == 6 && d[1] == 15 && d[2] == 24) << std::endl;

    short e[2][2] = {{1, -2}, {-3, 4}};
    short f[2];
    compute_row_sum<2, 2>(e, f);
    std::cout << (f[0] == -1 && f[1] == 1) << std::endl;

    return 0;
}
