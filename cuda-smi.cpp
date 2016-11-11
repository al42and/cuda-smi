/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2014 Andrey Alekseenko
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * 
 * */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#ifndef NO_NVML
#include "nvml.h"
#endif // def NO_NVML

#define CUDA_CALL(function, ...)  { \
    cudaError_t status = function(__VA_ARGS__); \
    anyCheck(status == cudaSuccess, cudaGetErrorString(status), #function, __FILE__, __LINE__); \
}

#define NVML_CALL(function, ...)  { \
    nvmlReturn_t status = function(__VA_ARGS__); \
    anyCheck(status == NVML_SUCCESS, nvmlErrorString(status), #function, __FILE__, __LINE__); \
}

void anyCheck(bool is_ok, const char *description, const char *function, const char *file, int line) {
    if (!is_ok) {
        fprintf(stderr,"Error: %s in %s at %s:%d\n", description, function, file, line);
        exit(EXIT_FAILURE);
    }
}

int main() {
    int cudaDeviceCount;
    struct cudaDeviceProp deviceProp;
    size_t memUsed, memTotal;
#ifndef NO_NVML
    unsigned int nvmlDeviceCount = 0;
    nvmlPciInfo_t nvmlPciInfo;
    nvmlMemory_t nvmlMemory;
    nvmlDevice_t nvmlDevice;
#endif // ndef NO_NVML

    CUDA_CALL(cudaGetDeviceCount, &cudaDeviceCount);
#ifndef NO_NVML
    NVML_CALL(nvmlInit);
    NVML_CALL(nvmlDeviceGetCount, &nvmlDeviceCount);
#endif // ndef NO_NVML

    for (int deviceId = 0; deviceId < cudaDeviceCount; ++deviceId) {
        CUDA_CALL(cudaGetDeviceProperties, &deviceProp, deviceId);
#ifdef NO_NVML
        printf("Device %2d", deviceId);
#else // def NO_NVML
        int nvmlDeviceId = -1;
        for (int nvmlId = 0; nvmlId < nvmlDeviceCount; ++nvmlId) {
            NVML_CALL(nvmlDeviceGetHandleByIndex, nvmlId, &nvmlDevice);
            NVML_CALL(nvmlDeviceGetPciInfo, nvmlDevice, &nvmlPciInfo);
            if (deviceProp.pciDomainID == nvmlPciInfo.domain &&
                deviceProp.pciBusID    == nvmlPciInfo.bus    &&
                deviceProp.pciDeviceID == nvmlPciInfo.device) {

                nvmlDeviceId = nvmlId;
                break;
            }
        }
        printf("Device %2d [nvidia-smi %2d]", deviceId, nvmlDeviceId);
#endif // def NO_NVML
        printf(" [PCIe %04x:%02x:%02x.0]", deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);
        printf(": %20s (CC %d.%d)", deviceProp.name, deviceProp.major, deviceProp.minor);
#ifdef NO_NVML
        CUDA_CALL(cudaSetDevice, deviceId);
        size_t memFree;
        CUDA_CALL(cudaMemGetInfo, &memFree, &memTotal);
        memUsed = memTotal - memFree;
        memUsed = memUsed / 1024 / 1024;
        memTotal = memTotal / 1024 / 1024;
#else
        if (nvmlDeviceId != -1) {
            NVML_CALL(nvmlDeviceGetMemoryInfo, nvmlDevice, &nvmlMemory);
            memUsed = nvmlMemory.used / 1024 / 1024;
            memTotal = nvmlMemory.total / 1024 / 1024;
        } else {
            memUsed = memTotal = 0;
        }
#endif // def NO_NVML
        printf(": %5zu of %5zu MiB Used", memUsed, memTotal);
        printf("\n");
    }
#ifndef NO_NVML
    NVML_CALL(nvmlShutdown);
#endif // ndef NO_NVML
    return 0;
}

