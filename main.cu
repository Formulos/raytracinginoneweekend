#include <iostream>
#include <fstream>
#include <curand_kernel.h>
#include <chrono>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"

using namespace std;

__device__ bool hit_sphere(const vec3& center, float radius, const ray& r) {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0 * dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - 4*a*c;
    return (discriminant > 0);
}

__device__ vec3 color(const ray& r, hitable **world) {
    hit_record rec;
    if ((*world)->hit(r, 0.0, MAXFLOAT, rec)) {
        return 0.5*vec3(rec.normal.x()+1, rec.normal.y()+1, rec.normal.z()+1);
    }
    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5*(unit_direction.y() + 1.0);
        return (1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
}

__global__ void render(vec3 *pixels,int nx,int ny,int ns,camera **cam,hitable **world){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= nx) || (j >= ny)) return;
    int pixel_index = j*nx + i;
    curandState rand_state;
    curand_init(1984+pixel_index, 0, 0, &rand_state);
    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&rand_state)) / float(nx);
        float v = float(j + curand_uniform(&rand_state)) / float(ny);
        ray r = (*cam)->get_ray(u,v);
        col += color(r, world);
    }
    pixels[pixel_index] = col/float(ns);
}
__global__ void create_world(hitable **d_list, hitable **d_world,camera **d_camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)   = new sphere(vec3(0,0,-1), 0.5);
        *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100);
        *d_world    = new hitable_list(d_list,2);
        *d_camera   = new camera();
    }
}
__global__ void free_world(hitable **d_list, hitable **d_world,camera **d_camera) {
    delete *(d_list);
    delete *(d_list+1);
    delete *d_world;
    delete *d_camera;
 }

void start() {
    using namespace std::chrono;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int nx = 1200;
    int ny = 800;
    int ns = 50;
    int pixel_block = 4;

    int pixels_size = nx*ny;
    size_t image_size = pixels_size*sizeof(vec3);

    vec3 *pixels;
    cudaMallocManaged((void **)&pixels, image_size);

    hitable **d_list;
    cudaMalloc((void **)&d_list, 2*sizeof(hitable *));
    hitable **d_world;
    cudaMalloc((void **)&d_world, sizeof(hitable *));
    camera **d_camera;
    cudaMalloc((void **)&d_camera, sizeof(camera *));
    create_world<<<1,1>>>(d_list,d_world,d_camera);
    cudaDeviceSynchronize();

    dim3 blocks(nx/pixel_block+1,ny/pixel_block+1);
    dim3 threads(pixel_block,pixel_block);

    render<<<blocks, threads>>>(pixels, nx, ny,  ns, d_camera, d_world);

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
    sfile.close();
    free_world<<<1,1>>>(d_list,d_world,d_camera);
    cudaFree(pixels);
    cudaFree(d_list);
    cudaFree(d_world);
    cudaFree(d_camera);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << time_span.count()<<std::endl;
}
int main(){
    int n = 1;
    for (int i=0; i<n;i++){
        start();
    }
}




