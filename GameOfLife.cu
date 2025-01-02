// Lab 5 Problem 1
// GameOfLife.cu

// compile using: nvcc -o GameOfLife.out GameOfLife.cu kernel.cu -lstdc++ -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
// run using: ./GameOfLife.out 1000 1000 5000
// can change the arguments, running arguments are: grid size x, grid size y, iterations

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono> 

// functions from the kernel file
void copy_opencv_to_device(const cv::Mat& mat, uchar*& dev_population, uchar*& dev_newpopulation);
void copy_device_to_opencv(const uchar* dev_population, cv::Mat& population);
__global__ void kernel(uchar* population, uchar* newpopulation, int ny, int nx);

// original code from Lecture 23 
#define MAX_SIZE 1024

int main(int argc, char** argv) {
    assert(argc == 4);

	//-----------------------
	// Convert Command Line
	//-----------------------
    
    int ny = atoi(argv[1]); //grid size x
    int nx = atoi(argv[2]); //grid size y
    int maxiter = atoi(argv[3]); //iterations

    assert(ny <= MAX_SIZE);
    assert(nx <= MAX_SIZE);

    //---------------------------------
    // Generate the initial image
    //---------------------------------
    srand(clock());
    cv::Mat population(ny, nx, CV_8UC1);
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            //seed a 1/2 density of alive (just arbitrary really)
            int state = rand() % 2;
            if (state == 0) population.at<uchar>(iy,ix) = 255; //dead
            else population.at<uchar>(iy,ix) = 0;   //alive
        }
    }

    // code modified here
    // allocate device memory once and copy the initial population
    uchar* dev_population;
    uchar* dev_newpopulation;
    copy_opencv_to_device(population, dev_population, dev_newpopulation);

    // Setup grid and block sizes for the kernel
    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);


    cv::Mat image_for_viewing(MAX_SIZE, MAX_SIZE, CV_8UC1);
    cv::namedWindow("Population", cv::WINDOW_AUTOSIZE);

    // start timing entire simulation
    auto start_time = std::chrono::high_resolution_clock::now(); // Start the timer

    for (int iter = 0; iter < maxiter; iter++) {
        // call CUDA kernel to update population
        kernel<<<grid, block>>>(dev_population, dev_newpopulation, ny, nx);
        
        // synchronize ensures the kernel has finished
        cudaDeviceSynchronize();

        // swap population and new population
        std::swap(dev_population, dev_newpopulation);

        // see the result every 10 iterations or frames
        // gets adjusted for question 4
        if (iter % 10 == 0) {
            copy_device_to_opencv(dev_population, population);
            //something new here - we will resize our image up to MAX_SIZE x MAX_SIZE so its not really tiny on the screen
            cv::resize(population, image_for_viewing, image_for_viewing.size(), cv::INTER_LINEAR);
            cv::imshow("Population", image_for_viewing);
            cv::waitKey(10);  //wait 10 seconds before closing image (or a keypress to close)
        }
    }

    // moved the juicy part of the code to kernel.cu

    // end timing of entire simulation
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Total simulation time: " << duration.count() << " seconds\n";

    // free up the device memory
    cudaFree(dev_population);
    cudaFree(dev_newpopulation);

    return 0;
}
