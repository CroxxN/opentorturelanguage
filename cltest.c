#define CL_TARGET_OPENCL_VERSION 300



#include "CL/cl_ext.h"
#include <stdint.h>
#include <stdlib.h>
#include "CL/cl.h"
#include "CL/opencl.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static const unsigned int VECSIZE=100;

void error(cl_int err, int section) {
    if (err != CL_SUCCESS) {
        printf("An error occured: %d\n on section %d", err, section);
        exit(1);
    }
}

char *load_kernel_from_file(char *file_name) {
    FILE *f = fopen(file_name, "rb");
    fseek(f, 0, SEEK_END);
    int size = ftell(f);
    assert(size > 0);
    rewind(f);
    char *kernel = malloc(size + 2); // +1 for the \0
    assert(kernel != NULL);
    fread(kernel, size, 1, f);
    kernel[size] = '\0';
    return kernel;
}

void fill_vec(float *vec) {
    for (unsigned int i = 0; i < VECSIZE; i++) {
        vec[i] = rand() / (float)RAND_MAX;
    }
}
void print_vec(float *vec) {
    for (unsigned int i = 0; i < VECSIZE; i++) {
        printf("%f ", vec[i]);
    }
}

int main() {

    cl_int err = CL_SUCCESS;
    cl_uint numplat = 0;
    cl_uint numdevices = 0;
    err = clGetPlatformIDs(0, NULL, &numplat);
    if (err == CL_SUCCESS)
        printf("%u devices found\n", numplat);
    else
        error(err, 1);
    cl_platform_id platforms[numplat];
    clGetPlatformIDs(numplat, platforms, NULL);
    cl_device_id device_id;
    for (cl_uint k = 0; k < numplat; k++) {
        err = clGetDeviceIDs(platforms[k], CL_DEVICE_TYPE_ALL, 0, NULL,
                             &numdevices);
        error(err, 2);
        printf("Found %u device[s]\n", numdevices);
        cl_device_id devices;
        err =
            clGetDeviceIDs(platforms[k], CL_DEVICE_TYPE_ALL, 1, &devices, NULL);
        error(err, 3);
        cl_char platform_name[1000];
        err = clGetPlatformInfo(platforms[k], CL_PLATFORM_NAME, 1000,
                                platform_name, NULL);
        error(err, 4);
        printf("%s\n", platform_name);
        cl_char string[100] = {0};
        err = clGetDeviceInfo(devices, CL_DEVICE_NAME, sizeof(string), string,
                              NULL);
        error(err, 5);
        printf("%s\n", string);
        cl_uint cum_unit; // compilation units
        err = clGetDeviceInfo(devices, CL_DEVICE_MAX_COMPUTE_UNITS,
                              sizeof(cl_uint), &cum_unit, NULL);
        error(err, 6);
        printf("There are %u compute units\n", cum_unit);
        cl_ulong global_max_memory;
        err = clGetDeviceInfo(devices, CL_DEVICE_GLOBAL_MEM_SIZE,
                              sizeof(cl_ulong), &global_max_memory, NULL);
        error(err, 7);
        printf("There is %lu gb global memory\n",
               global_max_memory / 1024 / 1024 / 1024);
        cl_ulong local_max_memory;
        err = clGetDeviceInfo(devices, CL_DEVICE_GLOBAL_MEM_SIZE,
                              sizeof(cl_ulong), &local_max_memory, NULL);
        error(err, 8);
        printf("There is %lu gb local memory\n",
               local_max_memory / 1024 / 1024 / 1024);
        err = clGetDeviceIDs(platforms[k], CL_DEVICE_TYPE_DEFAULT,
                             sizeof(cl_device_id), &device_id, NULL);
        error(err, 9);
        break;
    }
    cl_context ctx = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    error(err, 10);
    cl_command_queue cmd_queue =
        clCreateCommandQueueWithProperties(ctx, device_id, 0, &err);
    error(err, 11);
    char *program_source = load_kernel_from_file("vadkernel.cl");
    assert(program_source != NULL);
    size_t size = strlen(program_source);
    cl_program program = clCreateProgramWithSource(
        ctx, 1, (const char **)&program_source, &size, &err);
    error(err, 12);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    error(err, 13);
    cl_kernel vad_kernel = clCreateKernel(program, "vaad", &err);
    error(err, 14);

    float *a = malloc(sizeof(float)*VECSIZE); // vector a
    float *b= malloc(sizeof(float)*VECSIZE); // vector b
    float *c= malloc(sizeof(float)*VECSIZE); // resultant vector
    fill_vec(a);
    fill_vec(b);

    cl_mem v_a;
    cl_mem v_b;
    cl_mem v_c;
    v_a = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                         sizeof(float) * VECSIZE, NULL, &err);
    error(err, 15);
    v_b = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         sizeof(float) * VECSIZE, &b, &err);
    error(err, 16);
    v_c = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * VECSIZE, NULL,
                         &err);
    error(err, 17);

    err = clEnqueueWriteBuffer(cmd_queue, v_a, CL_TRUE, 0, sizeof(float)*VECSIZE, a, 0, NULL, NULL);
    error(err, 69);

    err = clEnqueueWriteBuffer(cmd_queue, v_b, CL_TRUE, 0, sizeof(float)*VECSIZE, b, 0, NULL, NULL);
    error(err, 70);

    err = clSetKernelArg(vad_kernel, 0, sizeof(cl_mem), &v_a);
    error(err, 18);
    err = clSetKernelArg(vad_kernel, 1, sizeof(cl_mem), &v_b);
    error(err, 19);
    err = clSetKernelArg(vad_kernel, 2, sizeof(cl_mem), &v_c);
    error(err, 20);
    err = clSetKernelArg(vad_kernel, 3, sizeof(unsigned int), &VECSIZE);
    error(err, 21);
    size_t global = VECSIZE;
    err = clEnqueueNDRangeKernel(cmd_queue, vad_kernel, 1, 0, &global, NULL, 0, NULL, NULL);
    error(err, 22);
    err = clFinish(cmd_queue);
    error(err, 23);

    clEnqueueReadBuffer(cmd_queue, v_c, CL_TRUE, 0, sizeof(float)*VECSIZE, v_c, 0, NULL, NULL);
    error(err, 24);
    print_vec(c);
    printf("\n\n");
    for(int i=0; i<VECSIZE; i++){
        printf("%f", a[i]+b[i]);
    }
    clReleaseMemObject(v_a);
    clReleaseMemObject(v_b);
    clReleaseMemObject(v_c);
    clReleaseProgram(program);
    clReleaseKernel(vad_kernel);
    clReleaseCommandQueue(cmd_queue);
    clReleaseContext(ctx);

    free(a);
    free(b);
    free(c);
    return 0;
}
