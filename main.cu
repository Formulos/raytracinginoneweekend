#include <iostream>
#include <fstream>

using namespace std;

__global__ void render(float *pixels,int nx,int ny){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= nx) || (j >= ny)) return;
    int pixel_index = j*nx*3 + i*3;
    fb[pixel_index + 0] = float(i) / nx;
    fb[pixel_index + 1] = float(j) / ny;
    fb[pixel_index + 2] = 0.2;
}

int main() {
    int nx = 200;
    int ny = 100;
    int pixel_block = 16

    int pixels_size = nx*ny;
    size_t image_size = 3*pixels_size*sizeof(float);

    float *pixels;
    cudaMallocManaged((void **)&pixels, pixels_size);

    dim3 blocks(nx/pixel_block+1,ny/pixel_block+1);
    dim3 threads(pixel_block,pixel_block);

    render<<<blocks, threads>>>(fb, nx, ny);

    cudaDeviceSynchronize();

    std::ofstream sfile;
    sfile.open("image", ios::out);
    sfile << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*3*nx + i*3;
            float r = fb[pixel_index + 0];
            float g = fb[pixel_index + 1];
            float b = fb[pixel_index + 2];
            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);
            sfile << ir << " " << ig << " " << ib << "\n";
        }
    }
}




