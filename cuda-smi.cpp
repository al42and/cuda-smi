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
#include "nvml.h"

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
    unsigned int nvmlDeviceCount = 0;
    struct cudaDeviceProp deviceProp;
    nvmlPciInfo_t nvmlPciInfo;
    nvmlMemory_t nvmlMemory;
    nvmlDevice_t nvmlDevice;
    size_t memUsed, memTotal;

    CUDA_CALL(cudaGetDeviceCount, &cudaDeviceCount);
    NVML_CALL(nvmlInit);
    NVML_CALL(nvmlDeviceGetCount, &nvmlDeviceCount);

    for (int deviceId = 0; deviceId < cudaDeviceCount; ++deviceId) {
        CUDA_CALL(cudaGetDeviceProperties, &deviceProp, deviceId);
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
        printf(" [PCIe %04x:%02x:%02x.0]", deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);
        printf(": %20s (CC %d.%d)", deviceProp.name, deviceProp.major, deviceProp.minor);
        if (nvmlDeviceId != -1) {
            NVML_CALL(nvmlDeviceGetMemoryInfo, nvmlDevice, &nvmlMemory);
            memUsed = nvmlMemory.used / 1024 / 1024;
            memTotal = nvmlMemory.total / 1024 / 1024;
        } else {
            memUsed = memTotal = 0;
        }
        printf(": %5zu of %5zu MiB Used", memUsed, memTotal);
        printf("\n");
    }
    NVML_CALL(nvmlShutdown);
    return 0;
}

