#include "CL/cl_ext.h"
#include <stdint.h>
#define CL_TARGET_OPENCL_VERSION 300
#include "CL/cl.h"
#include "CL/opencl.h"
#include <assert.h>
#include <stdio.h>
#include<unistd.h>
#include<string.h>

void error(cl_int err) {
    if (err != CL_SUCCESS) {
        printf("An error occured: %d\n", err);
        exit(1);
    }
}

char * load_kernel_from_file(char *file_name){
    FILE *f = fopen(file_name, "rb");
    fseek(f, 0, SEEK_END);
    int size = ftell(f);
    assert(size>0);
    rewind(f);
    char *kernel = malloc(size + 2); // +1 for the \0
    assert(kernel!=NULL);
    fread(kernel, size, 1, f);
    kernel[size] = '\0';
    return kernel;
}

int main() {

    cl_int err = CL_SUCCESS;
    cl_uint numplat = 0;
    cl_uint numdevices = 0;
    err = clGetPlatformIDs(0, NULL, &numplat);
    if (err == CL_SUCCESS)
        printf("%u devices found\n", numplat);
    else
        error(err);
    cl_platform_id platforms[numplat];
    clGetPlatformIDs(numplat, platforms, NULL);
    cl_device_id device_id;
    for (cl_uint k = 0; k < numplat; k++) {
        err = clGetDeviceIDs(platforms[k], CL_DEVICE_TYPE_ALL, 0, NULL,
                             &numdevices);
        error(err);
        printf("Found %u device[s]\n", numdevices);
        cl_device_id devices;
        err = clGetDeviceIDs(platforms[k], CL_DEVICE_TYPE_ALL, 1,
                             &devices, NULL);
        error(err);
        cl_char platform_name[1000];
        err = clGetPlatformInfo(platforms[k], CL_PLATFORM_NAME, 1000,
                                platform_name, NULL);
        error(err);
        printf("%s\n", platform_name);
        cl_char string[100] = {0};
        err = clGetDeviceInfo(devices, CL_DEVICE_NAME, sizeof(string),
                              string, NULL);
        error(err);
        printf("%s\n", string);
        cl_uint cum_unit; // compilation units
        err = clGetDeviceInfo(devices, CL_DEVICE_MAX_COMPUTE_UNITS,
                              sizeof(cl_uint), &cum_unit, NULL);
        error(err);
        printf("There are %u compute units\n", cum_unit);
        cl_ulong global_max_memory;
        err = clGetDeviceInfo(devices, CL_DEVICE_GLOBAL_MEM_SIZE,
                              sizeof(cl_ulong), &global_max_memory, NULL);
        error(err);
        printf("There is %lu gb global memory\n",
               global_max_memory / 1024 / 1024 / 1024);
        cl_ulong local_max_memory;
        err = clGetDeviceInfo(devices, CL_DEVICE_GLOBAL_MEM_SIZE,
                              sizeof(cl_ulong), &local_max_memory, NULL);
        error(err);
        printf("There is %lu gb global memory\n",
               local_max_memory / 1024 / 1024 / 1024);
        err = clGetDeviceIDs(platforms[k], CL_DEVICE_TYPE_DEFAULT,
                              sizeof(cl_device_id), &device_id, NULL);
        error(err);
        break;
    }
    cl_context ctx = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    error(err);
    cl_command_queue cmd_queue = clCreateCommandQueueWithProperties(ctx, device_id, 0, &err);
    error(err);
    char *program_source = load_kernel_from_file("vadkernel.cl");
    assert(program_source != NULL);
    cl_program program = clCreateProgramWithSource(ctx, 1, (const char **)&program_source, NULL, &err);
    error(err);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    error(err) ;
    cl_kernel vad_kernel = clCreateKernel(program, "vaad", &err);
    error(err);
    
    return 0;
}
