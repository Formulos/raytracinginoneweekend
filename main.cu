#include <iostream>
#include <fstream>
#include "vec3.h"
#include "ray.h"

using namespace std;

__device__ vec3 color(const ray& r) {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5*(unit_direction.y() + 1.0);
    return (1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}

__global__ void render(vec3 *pixels,int nx,int ny,vec3 lower_left_corner,vec3 horizontal,vec3 vertical,vec3 origin){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= nx) || (j >= ny)) return;
    int pixel_index = j*nx + i;
    float u = float(i) / float(nx);
    float v = float(j) / float(ny);
    ray r(origin, lower_left_corner + u*horizontal + v*vertical);
    pixels[pixel_index] = color(r);
}

int main() {
    int nx = 200;
    int ny = 100;
    int pixel_block = 16;

    int pixels_size = nx*ny;
    size_t image_size = pixels_size*sizeof(vec3);

    vec3 *pixels;
    cudaMallocManaged((void **)&pixels, image_size);

    dim3 blocks(nx/pixel_block+1,ny/pixel_block+1);
    dim3 threads(pixel_block,pixel_block);

    render<<<blocks, threads>>>(pixels, nx, ny,
        vec3(-2.0, -1.0, -1.0),
        vec3(4.0, 0.0, 0.0),
        vec3(0.0, 2.0, 0.0),
        vec3(0.0, 0.0, 0.0));

    cudaDeviceSynchronize();

    std::ofstream sfile;
    sfile.open("image", ios::out);
    sfile << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*pixels[pixel_index].r());
            int ig = int(255.99*pixels[pixel_index].g());
            int ib = int(255.99*pixels[pixel_index].b());
            sfile << ir << " " << ig << " " << ib << "\n";
        }
    }
}




