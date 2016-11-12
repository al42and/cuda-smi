/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Andrey Alekseenko, Kentaro Wada
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
#endif // ndef NO_NVML

#define CUDA_CALL(function, ...)  { \
    cudaError_t status = function(__VA_ARGS__); \
    anyCheck(status == cudaSuccess, cudaGetErrorString(status), #function, __FILE__, __LINE__); \
}

#ifndef NO_NVML
#define NVML_CALL(function, ...)  { \
    nvmlReturn_t status = function(__VA_ARGS__); \
    anyCheck(status == NVML_SUCCESS, nvmlErrorString(status), #function, __FILE__, __LINE__); \
}
#else // ndef NO_NVML
#define NVML_CALL(function, ...) { } // We create dummy wrapper to skip initialization code etc.
#endif // ndef NO_NVML

void anyCheck(bool is_ok, const char *description, const char *function, const char *file, int line) {
    if (!is_ok) {
        fprintf(stderr,"Error: %s in %s at %s:%d\n", description, function, file, line);
        exit(EXIT_FAILURE);
    }
}

void getMemoryUsageCUDA(int deviceId, size_t &memUsed, size_t &memTotal) {
    size_t memFree;
    CUDA_CALL(cudaSetDevice, deviceId);
    CUDA_CALL(cudaMemGetInfo, &memFree, &memTotal);
    memUsed = memTotal - memFree;
    memUsed = memUsed / 1024 / 1024;
    memTotal = memTotal / 1024 / 1024;
}

#ifndef NO_NVML
void getMemoryUsageNVML(nvmlDevice_t &nvmlDevice, size_t &memUsed, size_t &memTotal) {
    nvmlMemory_t nvmlMemory;
    NVML_CALL(nvmlDeviceGetMemoryInfo, nvmlDevice, &nvmlMemory);
    memUsed = nvmlMemory.used / 1024 / 1024;
    memTotal = nvmlMemory.total / 1024 / 1024;
}
#endif

int main() {
    int cudaDeviceCount;
    struct cudaDeviceProp deviceProp;
    size_t memUsed, memTotal;
#ifndef NO_NVML
    unsigned int nvmlDeviceCount;
    nvmlPciInfo_t nvmlPciInfo;
    nvmlDevice_t nvmlDevice;
#endif // ndef NO_NVML

    CUDA_CALL(cudaGetDeviceCount, &cudaDeviceCount);
    NVML_CALL(nvmlInit);
    NVML_CALL(nvmlDeviceGetCount, &nvmlDeviceCount);

    for (int deviceId = 0; deviceId < cudaDeviceCount; ++deviceId) {
        CUDA_CALL(cudaGetDeviceProperties, &deviceProp, deviceId);
        printf("Device %2d", deviceId);
#ifndef NO_NVML
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
        printf(" [nvidia-smi %2d]", nvmlDeviceId);
#endif // ndef NO_NVML

        printf(" [PCIe %04x:%02x:%02x.0]", deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);
        printf(": %20s (CC %d.%d)", deviceProp.name, deviceProp.major, deviceProp.minor);

#ifndef NO_NVML
        getMemoryUsageNVML(nvmlDevice, memUsed, memTotal);
#else //  ndef NO_NVML
        getMemoryUsageCUDA(deviceId, memUsed, memTotal);
#endif // ndef NO_NVML
        printf(": %5zu of %5zu MiB Used", memUsed, memTotal);
        printf("\n");
    }

    NVML_CALL(nvmlShutdown);
    return 0;
}

