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
#include <cuda_runtime_api.h>
#include "nvml.h"

int main(){
    int cudaDeviceCount;
    unsigned int nvmlDeviceCount = 0;
    struct cudaDeviceProp deviceProp;
    nvmlPciInfo_t nvmlPciInfo;
    nvmlMemory_t nvmlMemory;
    nvmlDevice_t nvmlDevice;
    size_t memUsed, memTotal;
    cudaGetDeviceCount(&cudaDeviceCount);
    nvmlInit();
    nvmlDeviceGetCount(&nvmlDeviceCount);
    for (int deviceId = 0; deviceId < cudaDeviceCount; ++deviceId) {
    	cudaGetDeviceProperties(&deviceProp, deviceId);
        int nvmlDeviceId = -1;
        for (int nvmlId = 0; nvmlId < nvmlDeviceCount; ++nvmlId) {
            nvmlDeviceGetHandleByIndex(nvmlId, &nvmlDevice);
            nvmlDeviceGetPciInfo(nvmlDevice, &nvmlPciInfo);
            if (deviceProp.pciDomainID == nvmlPciInfo.domain && 
                deviceProp.pciBusID    == nvmlPciInfo.bus    &&
                deviceProp.pciDeviceID == nvmlPciInfo.device) {
                nvmlDeviceId = nvmlId;
                break;
            }
        }
    	printf("Device %2d [nvidia-smi %2d]: %20s (CC %d.%d)", deviceId, nvmlDeviceId, deviceProp.name, deviceProp.major, deviceProp.minor);
        if (nvmlDeviceId != -1) {
            nvmlDeviceGetMemoryInfo(nvmlDevice, &nvmlMemory);
            memUsed = nvmlMemory.used / 1024 / 1024;
            memTotal = nvmlMemory.total / 1024 / 1024;
        	printf(": %5zu of %5zu MiB Used", memUsed, memTotal);
        } else {
            printf(": ??? MiB Used");
        }
    	printf(" [PCIe ID: %04x:%02x:%02x.0]", deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);
        printf("\n");
    }
    nvmlShutdown();
    return 0;
}

