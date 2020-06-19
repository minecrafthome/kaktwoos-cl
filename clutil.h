#ifndef __CLUTIL_H
#define __CLUTIL_H

#include <stdio.h>
#include <CL/cl.h>

#ifndef CLUTIL_NAME_LEN
#define CLUTIL_NAME_LEN 100
#endif

const char *getDeviceTypeString(cl_device_type type) {
  switch (type) {
    case CL_DEVICE_TYPE_CPU: return "CPU";
    case CL_DEVICE_TYPE_GPU: return "GPU";
    case CL_DEVICE_TYPE_ACCELERATOR: return "Accelerator";
    case CL_DEVICE_TYPE_CUSTOM: return "Custom";
    default: return "Unknown";
  }
}

typedef struct _device_info {
  cl_device_type type;
  char vendor[CLUTIL_NAME_LEN];
  char name[CLUTIL_NAME_LEN];
  char version[CLUTIL_NAME_LEN];
  cl_uint compute_units;
  char *info_str;
} device_info;

device_info *getDeviceInfo(cl_device_id id) {
  device_info *dev_info = malloc(sizeof(device_info));
  clGetDeviceInfo(id, CL_DEVICE_TYPE, sizeof(cl_device_type), &dev_info->type, NULL);
  clGetDeviceInfo(id, CL_DEVICE_VENDOR, CLUTIL_NAME_LEN, &dev_info->vendor, NULL);
  clGetDeviceInfo(id, CL_DEVICE_NAME, CLUTIL_NAME_LEN, &dev_info->name, NULL);
  clGetDeviceInfo(id, CL_DEVICE_VERSION, CLUTIL_NAME_LEN, &dev_info->version, NULL);
  clGetDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &dev_info->compute_units, NULL);
  size_t len = (CLUTIL_NAME_LEN + 1) * 3 + 20;
  dev_info->info_str = malloc(len);
  snprintf(dev_info->info_str, len, "%s %s %s %s, %u CUs",
    getDeviceTypeString(dev_info->type),
    dev_info->vendor,
    dev_info->name,
    dev_info->version,
    dev_info->compute_units
  );
  return dev_info;
}

char *getPlatformInfo(cl_platform_id id) {
  char vendor[CLUTIL_NAME_LEN];
  clGetPlatformInfo(id, CL_PLATFORM_VENDOR, CLUTIL_NAME_LEN, vendor, NULL);
  char name[CLUTIL_NAME_LEN];
  clGetPlatformInfo(id, CL_PLATFORM_NAME, CLUTIL_NAME_LEN, name, NULL);
  char version[CLUTIL_NAME_LEN];
  clGetPlatformInfo(id, CL_PLATFORM_VERSION, CLUTIL_NAME_LEN, version, NULL);
  char profile[CLUTIL_NAME_LEN];
  clGetPlatformInfo(id, CL_PLATFORM_PROFILE, CLUTIL_NAME_LEN, profile, NULL);
  char extensions[CLUTIL_NAME_LEN];
  clGetPlatformInfo(id, CL_PLATFORM_EXTENSIONS, CLUTIL_NAME_LEN, extensions, NULL);
  size_t len = (CLUTIL_NAME_LEN + 1) * 5 + 20;
  char *info = malloc(len);
  snprintf(info, len, "%s %s %s %s %s", vendor, name, version, profile, extensions);
  return info;
}

const char *getErrorString(cl_int error)
{
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

static inline void check(cl_int code, char *prefix) {
  if (code == CL_SUCCESS) return;
  fprintf(stderr, "%s: %s\n", prefix ? prefix : "An OpenCL error occured", getErrorString(code));
  exit(code);
}

static inline char *readFile(const char *name) {
    FILE *fp;
    char *source_str;
    size_t program_size;

    fp = fopen(name, "rb");
    if (!fp) {
        printf("Failed to load kernel\n");
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    program_size = ftell(fp);
    rewind(fp);
    source_str = (char*)malloc(program_size + 1);
    source_str[program_size] = '\0';
    fread(source_str, sizeof(char), program_size, fp);
    fclose(fp);
    return source_str;
}

static inline char *getBuildLog(cl_program program, cl_device_id device) {
  size_t log_size;
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
  char *log = (char *) malloc(log_size);
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
  return log;
}

static inline char *getLog(cl_program program, cl_device_id device) {
  size_t log_size;
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
  char *log = (char *) malloc(log_size);
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
  return log;
}

#endif
