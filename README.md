# Game-Of-Life
This project is a CUDA implementation of the Game of Life.

The kernel function is responsible for updating the population grid based on the Game of Life rules. For this instance, a cell is alive if it has exactly 2 or 3 alive neighbours. A dead cell becomes alive if it has exactly 3 neighbours. Our kernel function was written in a separate CUDA file and is attached. The input parameters are the population (current state of grid), the new population (new grid to store), hight and width of grid.

We first calculate the indices of the current thread in the grid which allow us to map threads to the grid elements. If check boundaries if these indices go out of bounds then we return. We then check all 8 neighbours of the current cell which includes the wraparound conditions. These states and count is used to determine the next state of the current cell based on the Game of Life’s rules. If the cell is already alive (indicated as 0), it stays alive if it has 2 or 3 alive neighbours. It otherwise dies. If the cell is dead (indicated by 255), it comes to life if it has 3 alive neighbours. The new state is the stored in new population. 

Back in the main game of life CUDA file, we launch the kernel by giving the grid size and block size of 16x16 threads ensuring 256 threads per block. The grid is divided into blocks based on the size of the grid and the block size. The total number of threads launched is equal to the number of cells in the grid.

Below is the visual output for a grid size of 200 by 200 for 5000 frames.

![pic2](https://github.com/user-attachments/assets/20a74d7a-e217-4ce6-8046-85ed6f538524)
![pic3](https://github.com/user-attachments/assets/94219fee-c9fe-4839-9766-2a17a33ae585)

