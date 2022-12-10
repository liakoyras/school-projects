#include <iostream>
#include <cstdlib>
#include <ctime>

#include "ac_int.h"
#include "ac_math.h"

#include "types.h"

/**
 * Perform one iteration of the K-Means algorithm.
 *
 * The samples are clustered to the nearest center using Manhattan distance
 * and then calculate the new centers using the mean of the points clustered.
 * If the centers did not change through an iteration (output centers are the
 * same as input centers), the algorithm has converged.
 * 
 * @tparam N The number of samples that will be clustered.
 * @tparam M The number of clusters (k) that K-Means will group samles to.
 *
 * @param[in] points The coordinates of the data samples.
 * @param[in, out] ID The label of the cluster each sample belongs to.
 * @param[in, out] center The coordinates of the centers 
 *
 * @returns true if the algorithm converged.
 */
template<int N, int M>
bool kmeans(Point points[N], int ID[N], Point center[M]) {
    POINTS: for (int i = 0; i < N; i++) {
        // Read each data sample
        ac_int<16, false> x = points[i].x;
        ac_int<16, false> y = points[i].y;
        
        ac_int<17, false> min_distance = -1; // Initialize to the max value
        DIST: for (int j=0; j < M; j++) {
            // Read each center
            ac_int<16, false> center_x = center[j].x;
            ac_int<16, false> center_y = center[j].y;
            
            // Define the absolute value of each coordinate and the distance
            // The size is 17 bits because the max distance is at most the sum
            // of the two 16-bit coordinates (2 * (2^16 - 1)) (for points at
            // the top-left and bottom right corner)
            ac_int<16, false> abs_x, abs_y;
            ac_int<17, false> distance;
            
            // Calculate Manhattan distance
            ac_math::ac_abs(x-center_x, abs_x);
            ac_math::ac_abs(y-center_y, abs_y);
            distance = abs_x + abs_y;
            
            // Find the closest center and update ID[i]
            if (distance < min_distance) {
                min_distance = distance;
                ID[i] = j;
            }
        }
    }
    // Update the centers to the mean of cluster sampes based on the new
    // ID array.
    bool change = false; // Becomes true if the centers did change
    NEW_CENT: for (int j = 0; j < M; j++) { // Iterate through clusters
        ac_int<32, false> count = 0; // If all elements are in the same cluster
        ac_int<16, false> new_x, new_y;
        ac_int<48, false> sum_x = 0, sum_y = 0; // 2^32 samples * 2^16 value
        SUMS: for (int i = 0; i < N; i++) { // Iterate through samples
            if (ID[i] == j) { // Check if the sample belongs to current cluster
                // Find the sum of the points in the cluster
                count++;
                sum_x += points[i].x;
                sum_y += points[i].y;
            }
        }
        // Set new center coordinates at the mean value of cluster samples
        ac_math::ac_div(sum_x, count, new_x);
        ac_math::ac_div(sum_y, count, new_y);
        
        // Calculate absolute difference to check for convergence
        ac_int<16, false> diff_x, diff_y;
        ac_math::ac_abs(center[j].x-new_x, diff_x);
        ac_math::ac_abs(center[j].y-new_y, diff_y);
        if (diff_x >= 1 || diff_y >= 1) {
            // If the centers changed, update to the new center
            change = true;
            center[j].x = new_x;
            center[j].y = new_y;
        }
    }
    // If new and old centers are the same, return false, else return true
    return change;
}

/**
 * Program testbench.
 */
int main(int argc, char** argv) {
    std::srand(std::time(NULL)); // Use current time as seed for random generator
    const unsigned short N = 50; // Number of samples
    const unsigned short M = 3;  // Number of clusters
        
    // Randomly initialize data samples
    std::cout << "Samples:" << std::endl;
    Point points[N];
    for(int i = 0; i < N; i++) {
        Point p = {std::rand(), std::rand()};
        points[i] = p;
        std::cout << p.x << " " << p.y << std::endl;
    }
    
    // Randomly initialize centroids.
    std::cout << "The randomly chosen initial centers are:" << std::endl;
    Point center[M];
    for(int i = 0; i < M; i++) {
        Point p = {std::rand(), std::rand()};
        center[i] = p;
        std::cout << p.x << " " << p.y << std::endl;
    }
    
    // Dummy initial ID values (cluster labels)
    int ID[N] = {0};
    
    // Execute K-Means
    int iterations = 0;
    bool go_on = true;
    while (go_on) {
        iterations++;
        go_on = kmeans<N, M>(points, ID, center);
        std::cout << "Iteration " << iterations << ":" << std::endl;
        std::cout << "The centers are:" << std::endl;
        for (int i = 0; i < M; i++) {
            std::cout << center[i].x << " " << center[i].y << std::endl;
        }
    }
    
    // Final cluster IDs
    std::cout << "Cluster ID for each sample:" << std::endl;
    for (int i=0; i<N; i++) {
        std::cout << ID[i] << ", ";
    }
    std::cout << std::endl;
    
    std::cout << "The " << M << " centers are:" << std::endl;
    for (int i = 0; i < M; i++) {
        std::cout << center[i].x << " " << center[i].y << std::endl;
    }
    
    return 0;
}

