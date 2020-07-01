#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <string.h>
#include <time.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <stdbool.h>

#define CL_TARGET_OPENCL_VERSION 200

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <OpenCL/cl_platform.h>
#else
#include <CL/cl.h>
#include <CL/cl_platform.h>
#endif
#include "clutil.h"

#ifdef _WIN64
#include "boinc_win.h"
#else
#ifdef _WIN32
#include "boinc_win.h"
#endif
#endif

#include "boinc_api.h"

#define KERNEL_BUFFER_SIZE (0x4000)
#define MAX_SEED_BUFFER_SIZE (0x10000)

int main(int argc, char *argv[])
{
BOINC_OPTIONS options;

boinc_options_defaults(options);
options.normal_thread_priority = true;
boinc_init_options(&options);

    //boinc_init();

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
    size_t seedbuffer_size;

    struct checkpoint_vars {
        cl_ulong offset;
        cl_ulong start;
        int block;
	    double elapsed_chkpoint;
        int total_seed_count;
	};

    if (argc % 2 != 1) {
        printf("Failed to parse arguments\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 1; i < argc; i += 2) {
        const char *param = argv[i];
        if (strcmp(param, "-d") == 0 || strcmp(param, "--device") == 0) {
            gpuIndex = atoi(argv[i + 1]);
        } else if (strcmp(param, "-s") == 0 || strcmp(param, "--start") == 0) {
            sscanf(argv[i + 1], "%" SCNd64, &start);
        } else if (strcmp(param, "-e") == 0 || strcmp(param, "--end") == 0) {
            sscanf(argv[i + 1], "%" SCNd64, &end);
        } else if (strcmp(param, "-cs") == 0 || strcmp(param, "--chunkseed") == 0) {
            sscanf(argv[i + 1], "%" SCNd64, &chunkSeed);
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

    fprintf(stderr,"Received work unit: %" SCNd64 "\n", chunkSeed);
    fprintf(stderr,"Data: n1: %d, n2: %d, n3: %d, di: %d, ch: %d\n",
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

	fflush(stderr);

    FILE *kernel_file = boinc_fopen("kaktwoos.cl", "r");
    if (!kernel_file) {
        printf("Failed to open kernel");
        exit(1);
    }

    char *kernel_src = (char *)malloc(KERNEL_BUFFER_SIZE);
    size_t kernel_length = fread(kernel_src, 1, KERNEL_BUFFER_SIZE, kernel_file);

    fclose(kernel_file);

	APP_INIT_DATA aid;
	boinc_get_init_data(aid);

	if (aid.gpu_device_num >= 0) {
	 gpuIndex = aid.gpu_device_num;
	 fprintf(stderr,"boinc gpu %i gpuindex: %i \n", aid.gpu_device_num, gpuIndex);
	}
	else
	{
	fprintf(stderr,"stdalone gpuindex % \n", gpuIndex);
	}

    cl_platform_id platform_id = NULL;
    cl_device_id *device_ids;
    cl_uint num_devices;
    cl_uint num_platforms;
    cl_int err;

    check(clGetPlatformIDs(1, &platform_id, &num_platforms), "clGetPlatformIDs ");
    check(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices), "clGetDeviceIDs ");

	device_ids = (_cl_device_id**)malloc(num_devices * sizeof(*device_ids));
    check(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, num_devices, device_ids, &num_devices), "clGetDeviceIDs ");

        cl_context context = clCreateContext(NULL, 1, device_ids + gpuIndex, NULL, NULL, &err);
    check(err, "clCreateContext ");

        cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_ids[gpuIndex], NULL, &err);
    check(err, "clCreateCommandQueueWithProperties ");

    seedbuffer_size = 0x40 * sizeof(cl_ulong);

    // 16 Kb of memory for seeds
    cl_mem seeds = clCreateBuffer(context, CL_MEM_READ_WRITE, seedbuffer_size , NULL, &err);
    check(err, "clCreateBuffer (seeds) ");
    cl_mem data =  clCreateBuffer(context, CL_MEM_READ_ONLY, 10 * sizeof(int), NULL, &err);
    check(err, "clCreateBuffer (data) ");

    cl_program program = clCreateProgramWithSource(
            context,
            1,
            (const char **)&kernel_src,
            &kernel_length,
            &err);
    check(err, "clCreateProgramWithSource ");
    err = clBuildProgram(program, 1, device_ids + gpuIndex, NULL, NULL, NULL);

    if (err != CL_SUCCESS) {
        size_t len;
        clGetProgramBuildInfo(program, device_ids[gpuIndex], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);

        char *info = (char *)malloc(len);
        clGetProgramBuildInfo(program, device_ids[gpuIndex], CL_PROGRAM_BUILD_LOG, len, info, NULL);
        printf("%s\n", info);
        free(info);
    }

    cl_kernel kernel = clCreateKernel(program, "crack", &err);
    check(err, "clCreateKernel ");

    check(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&data), "clSetKernelArg (0) ");
    check(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&seeds), "clSetKernelArg (1) ");

    size_t work_unit_size = 4194304;
    size_t block_size = 64;

    arguments[1] = work_unit_size;

    cl_ulong offset = start;
    int block = 0;
    int total_seed_count = 0;
    int chkpoint_ready = 0;
    double seedrange = (end - start);

    cl_ulong found_seeds[MAX_SEED_BUFFER_SIZE];

    clock_t start_time, end_time, elapsed_chkpoint;
    start_time = clock();

    FILE *checkpoint_data = boinc_fopen("kaktpoint.txt", "rb");

    if (!checkpoint_data) {
     fprintf(stderr, "No checkpoint to load\n");
     }
     else {

	    void boinc_begin_critical_section();
        struct checkpoint_vars data_store;

	fread(&data_store, sizeof(data_store), 1, checkpoint_data);
        offset = data_store.offset;
	start = data_store.start;
        block = data_store.block;
	elapsed_chkpoint = data_store.elapsed_chkpoint;
        total_seed_count = data_store.total_seed_count;

        fread(found_seeds, sizeof(cl_ulong), total_seed_count, checkpoint_data);
	
	fprintf(stderr, "Checkpoint loaded, task time %d s \n", elapsed_chkpoint);
        fclose(checkpoint_data);
	void boinc_end_critical_section();
    }

    while (offset < end) {

        arguments[0] =  block + start / work_unit_size;

        check(clEnqueueWriteBuffer(command_queue, data, CL_TRUE, 0, 10 * sizeof(int), arguments, 0, NULL, NULL), "clEnqueueWriteBuffer ");
        check(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &work_unit_size, &block_size, 0, NULL, NULL), "clEnqueueNDRangeKernel ");

        int *data_out = (int *)malloc(sizeof(int) * 10);
        check(clEnqueueReadBuffer(command_queue, data, CL_TRUE, 0, sizeof(int) * 10, data_out, 0, NULL, NULL), "clEnqueueReadBuffer (data) ");

        int seed_count = data_out[2];
        seedbuffer_size = sizeof(cl_ulong) + sizeof(cl_ulong) * seed_count;

        cl_ulong *result = (cl_ulong *)malloc(sizeof(cl_ulong) + sizeof(cl_ulong) * seed_count);
	    check(clEnqueueReadBuffer(command_queue, seeds, CL_TRUE, 0, seedbuffer_size, result, 0, NULL, NULL), "clEnqueueReadBuffer (seeds) ");

	end_time = clock();

        for (int i = 0; i < seed_count; i++) {
            fprintf(stderr,"    Found seed: %" SCNd64 ", %llu, height: %d\n",
                    result[i],
                    result[i] & ((1ULL << 48ULL) - 1ULL),
                    (int)(result[i] >> 58ULL));

            fprintf(stderr, "%" SCNd64 "\n", (cl_ulong)result[i]);
            found_seeds[total_seed_count++] = result[i];
	    }

        double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        offset += work_unit_size;
        block++;
        chkpoint_ready++;

	int boinc_time_to_checkpoint();
        if (chkpoint_ready >= 500 || boinc_time_to_checkpoint() ){  // 500 for 0.5bil seeds before checkpoint

           void boinc_begin_critical_section(); // Boinc should not interrupt this

	   boinc_delete_file("kaktpoint.txt");
           FILE *checkpoint_data = boinc_fopen("kaktpoint.txt", "wb");

            struct checkpoint_vars data_store;
            data_store.offset = offset;
	    data_store.start = start;
            data_store.block = block;
	    data_store.elapsed_chkpoint = (elapsed_chkpoint + (double)(end_time - start_time) / CLOCKS_PER_SEC);
            data_store.total_seed_count = total_seed_count;

            fwrite(&data_store, sizeof(data_store), 1, checkpoint_data);
            fwrite(found_seeds, sizeof(cl_ulong), total_seed_count, checkpoint_data);

            chkpoint_ready = 0;
	    fclose(checkpoint_data);

            double fraction_done = ((offset - start) / (seedrange));
            boinc_fraction_done(fraction_done);

	    void boinc_end_critical_section();
	    boinc_checkpoint_completed(); // Checkpointing completed
         }

        free(result);
        free(data_out);

    } // End of seed feed and processing loop

    void boinc_begin_critical_section();

    double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    fprintf(stderr, "Speed: %.2fm/s \n", (offset - start) / (elapsed_chkpoint + elapsed) / 1000000);

    fprintf(stderr, "Done\n");
    fprintf(stderr, "Processed %"SCNd64 " seeds in %f seconds\n",
            end - start,
            elapsed_chkpoint + ((double)(end_time - start_time) / CLOCKS_PER_SEC));

    fprintf(stderr,"Found seeds: \n");

   for (int i = 0; i < total_seed_count; i++) {
        fprintf(stderr,"    %" SCNd64 "\n", found_seeds[i]);
    }

    boinc_delete_file("kaktpoint.txt");
    check(clFlush(command_queue), "clFlush ");
    check(clFinish(command_queue), "clFinish ");
    check(clReleaseKernel(kernel), "clReleaseKernel ");
    check(clReleaseProgram(program), "clReleaseProgram ");
    check(clReleaseMemObject(seeds), "clReleaseMemObject (seeds) ");
    check(clReleaseMemObject(data), "clReleaseMemObject (data) ");
    check(clReleaseCommandQueue(command_queue), "clReleaseCommandQueue ");
    check(clReleaseContext(context), "clReleaseContext ");

    fflush(stderr);
    void boinc_end_critical_section();
    boinc_finish(0);

}
