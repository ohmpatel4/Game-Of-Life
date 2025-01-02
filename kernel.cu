// Lab 5 Problem 1
// kernel.cu

#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>

__global__ void kernel(uchar* population, uchar* newpopulation, int ny, int nx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // make sure we are in bounds
    if (idx >= nx || idy >= ny) return; 

    int occupied_neighbours = 0;

    // check all the 8 neighboring cells as well as neighbours wrapping around
    for (int jy = idy - 1; jy <= idy + 1; jy++) {
        for (int jx = idx - 1; jx <= idx + 1; jx++) {
            if (jx == idx && jy == idy) continue; 

            // wrap around conditions
            int row = (jy + ny) % ny;
            int col = (jx + nx) % nx;

            if (population[row * nx + col] == 0) {
                occupied_neighbours++;
            }
        }
    }

    // rules applied here
    if (population[idy * nx + idx] == 0) {
        if (occupied_neighbours == 2 || occupied_neighbours == 3)
            newpopulation[idy * nx + idx] = 0;  // alive
        else
            newpopulation[idy * nx + idx] = 255;  // dead
    } else {
        if (occupied_neighbours == 3)
            newpopulation[idy * nx + idx] = 0;  // reproduction
        else
            newpopulation[idy * nx + idx] = 255;  // dies
    }
}

// function to allocate memory on GPU
//function copy data from host to device
void copy_opencv_to_device(const cv::Mat& mat, uchar*& dev_population, uchar*& dev_newpopulation) {
    int size = mat.rows * mat.cols * sizeof(uchar);
    cudaMalloc(&dev_population, size);
    cudaMalloc(&dev_newpopulation, size);
    cudaMemcpy(dev_population, mat.data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_newpopulation, mat.data, size, cudaMemcpyHostToDevice);
}

// function to copy back data from device to host
void copy_device_to_opencv(const uchar* dev_population, cv::Mat& population) {
    cudaMemcpy(population.data, dev_population, population.rows * population.cols * sizeof(uchar), cudaMemcpyDeviceToHost);
}
