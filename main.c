#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <string.h>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 220

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <OpenCL/cl_platform.h>
#else
#include <CL/cl.h>
#include <CL/cl_platform.h>
#endif

#define KERNEL_BUFFER_SIZE (0x4000)
#define MAX_SEED_BUFFER_SIZE (0x10000)
#define PRINT_DELTA (3.0)

#ifdef DEBUG
#define checkErr(err) if (err != CL_SUCCESS) { printf("Error: %d at line %d\n", err, __LINE__); exit(EXIT_FAILURE); }
#else
#define checkErr(err) err
#endif

int main(int argc, char *argv[])
{
    int gpuIndex = 0;
    cl_ulong start = 0;
    cl_ulong end = 0;
    cl_ulong chunkSeed = 0;
    int chunkSeedBottom4Bits = 0;
    int chunkSeedBit5 = 0;
    int neighbor1 = 0;
    int neighbor2 = 0;
    int neighbor3 = 0;
    int diagonalIndex = 0;
    int cactusHeight = 0;

    char *strend;

    if (argc % 2 != 1) {
        printf("Failed to parse arguments\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 1; i < argc; i += 2) {
        const char *param = argv[i];
        if (strcmp(param, "-d") == 0 || strcmp(param, "--device") == 0) {
            gpuIndex = atoi(argv[i + 1]);
        } else if (strcmp(param, "-s") == 0 || strcmp(param, "--start") == 0) {
            start = strtoull(argv[i + 1], &strend, 10);
        } else if (strcmp(param, "-e") == 0 || strcmp(param, "--end") == 0) {
            end = strtoull(argv[i + 1], &strend, 10);
        } else if (strcmp(param, "-cs") == 0 || strcmp(param, "--chunkseed") == 0) {
            chunkSeed = strtoull(argv[i + 1], &strend, 10);
            chunkSeedBottom4Bits = (int)(chunkSeed & 15U);
            chunkSeedBit5 = (int)((chunkSeed >> 4U) & 1U);
        } else if (strcmp(param, "-n1") == 0 || strcmp(param, "--neighbor1") == 0) {
            neighbor1 = atoi(argv[i + 1]);
        } else if (strcmp(param, "-n2") == 0 || strcmp(param, "--neighbor2") == 0) {
            neighbor2 = atoi(argv[i + 1]);
        } else if (strcmp(param, "-n3") == 0 || strcmp(param, "--neighbor3") == 0) {
            neighbor3 = atoi(argv[i + 1]);
        } else if (strcmp(param, "-di") == 0 || strcmp(param, "--diagonalindex") == 0) {
            diagonalIndex = atoi(argv[i + 1]);
        } else if (strcmp(param, "-ch") == 0 || strcmp(param, "--cactusheight") == 0) {
            cactusHeight = atoi(argv[i + 1]);
        } else {
            printf("Unknown parameter: %s\n", param);
        }
    }

    printf("Received work unit: %lu\n", chunkSeed);
    printf("Data: n1: %d, n2: %d, n3: %d, di: %d, ch: %d\n",
        neighbor1,
        neighbor2,
        neighbor3,
        diagonalIndex,
        cactusHeight);

    int arguments[10] = {
        0,
        0,
        0,
        neighbor1,
        neighbor2,
        neighbor3,
        diagonalIndex,
        cactusHeight,
        chunkSeedBottom4Bits,
        chunkSeedBit5
    };
    FILE *kernel_file = fopen("kaktwoos.cl", "r");

    if (!kernel_file) {
        printf("Failed to open kernel");
        exit(1);
    }

    char *kernel_src = malloc(KERNEL_BUFFER_SIZE);
    size_t kernel_length = fread(kernel_src, 1, KERNEL_BUFFER_SIZE, kernel_file);

    fclose(kernel_file);

    cl_platform_id platform_id = NULL;
    cl_device_id device_ids[1];
    cl_uint num_devices;
    cl_uint num_platforms;
    cl_int err;
    checkErr(clGetPlatformIDs(1, &platform_id, &num_platforms));
    checkErr(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, device_ids, &num_devices));

    cl_context context = clCreateContext(NULL, 1, device_ids, NULL, NULL, &err);
    checkErr(err);
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_ids[0], NULL, &err);
    checkErr(err);

    // 16 Kb of memory for seeds
    cl_mem seeds = clCreateBuffer(context, CL_MEM_READ_WRITE, 0x40 * sizeof(unsigned long), NULL, &err);
    checkErr(err);
    cl_mem data =  clCreateBuffer(context, CL_MEM_READ_ONLY, 10 * sizeof(int), arguments, &err);
    checkErr(err);

    cl_program program = clCreateProgramWithSource(
            context,
            1,
            (const char **)&kernel_src,
            &kernel_length,
            &err);
    checkErr(err);
    err = clBuildProgram(program, 1, device_ids, NULL, NULL, NULL);

    if (err != CL_SUCCESS) {
        size_t len;
        clGetProgramBuildInfo(program, device_ids[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);

        char *info = malloc(len);
        clGetProgramBuildInfo(program, device_ids[0], CL_PROGRAM_BUILD_LOG, len, info, NULL);
        printf("%s\n", info);
        free(info);
    }

    cl_kernel kernel = clCreateKernel(program, "crack", &err);
    checkErr(err);

    checkErr(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&data));
    checkErr(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&seeds));

    size_t work_unit_size = 1048576;
    size_t block_size = 256;
    cl_ulong offset = start;
    int block = 0;
    arguments[1] = work_unit_size;

    clock_t start_time, end_time;

    cl_ulong found_seeds[MAX_SEED_BUFFER_SIZE];
    int total_seed_count = 0;
    double last_print = 0;

    start_time = clock();
    while (offset < end) {
        arguments[0] = block;
        checkErr(clEnqueueWriteBuffer(command_queue, data, CL_TRUE, 0, 10 * sizeof(int), arguments, 0, NULL, NULL));
        checkErr(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &work_unit_size, &block_size, 0, NULL, NULL));

        int *data_out = malloc(sizeof(int) * 10);
        checkErr(clEnqueueReadBuffer(command_queue, data, CL_TRUE, 0, sizeof(int) * 10, data_out, 0, NULL, NULL));
        int seed_count = data_out[2];
        cl_ulong *result = malloc(sizeof(cl_ulong) * seed_count);
        checkErr(clEnqueueReadBuffer(command_queue, seeds, CL_TRUE, 0, sizeof(cl_ulong) * seed_count, result, 0, NULL, NULL));
        end_time = clock();

        for (int i = 0; i < seed_count; i++) {
            printf("    Found seed: %lu, %llu, height: %d\n",
                    result[i],
                    result[i] & ((1ULL << 48ULL) - 1ULL),
                    (int)(result[i] >> 58ULL));
            fprintf(stderr, "%lu\n", (cl_ulong)result[i]);
            found_seeds[total_seed_count++] = result[i];
        }

        double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        if (elapsed - last_print > PRINT_DELTA) {
            printf("Speed: %.2fm/s, %.2f%%, %.1fs\n",
                (offset - start) / elapsed / 1000000,
                (double)(offset - start) / (end - start) * 100,
                elapsed);
            last_print = elapsed;
        }

        offset += work_unit_size;
        block++;
        free(result);
        free(data_out);
    }

    printf("Done\n");
    printf("Processed %lu seeds in %f seconds\n",
            end - start,
            (double)(end_time - start_time) / CLOCKS_PER_SEC);
    printf("Found seeds: \n");
    for (int i = 0; i < total_seed_count; i++) {
        printf("    %lu\n", found_seeds[i]);
    }

    checkErr(clFlush(command_queue));
    checkErr(clFinish(command_queue));
    checkErr(clReleaseKernel(kernel));
    checkErr(clReleaseProgram(program));
    checkErr(clReleaseMemObject(seeds));
    checkErr(clReleaseMemObject(data));
    checkErr(clReleaseCommandQueue(command_queue));
    checkErr(clReleaseContext(context));
    return 0;

}
