cuda-smi
========

A simple utility to show nVidia GPU memory usage.
Unlike `nvidia-smi`, it uses CUDA device IDs.

For a number of reasons nVidia uses different device enumeration in `nvidia-smi` monitoring utility and in their CUDA API, making it extremely frustrating to choose vacant GPU for calculations on multi-GPU machine.
This utility was made to solve this problem.

Code is distributed under MIT license, except `nvml.h` header which is property of NVIDIA Corporation.

Building
--------

The code is compiled statically to simplify distribution over a large number of machines.

Simply install more-or-less recent CUDA Toolkit and run `make`.

Output example
--------------
    aland@NX8-1:~$ cuda-smi 
    Device  0 [nvidia-smi  2]:      GeForce GTX 680 (CC 3.0):     9 of  2047 MiB Used [PCIe ID: 0000:13:00.0]
    Device  1 [nvidia-smi  3]:          Tesla C1060 (CC 1.3):     3 of  4095 MiB Used [PCIe ID: 0000:14:00.0]
    Device  2 [nvidia-smi  1]:          Tesla C1060 (CC 1.3):   106 of  4095 MiB Used [PCIe ID: 0000:0d:00.0]
    Device  3 [nvidia-smi  0]:          Tesla C2075 (CC 2.0):    13 of  6143 MiB Used [PCIe ID: 0000:0c:00.0]
    Device  4 [nvidia-smi  7]:          Tesla C1060 (CC 1.3):   106 of  4095 MiB Used [PCIe ID: 0000:8e:00.0]
    Device  5 [nvidia-smi  6]:          Tesla C2075 (CC 2.0):   115 of  6143 MiB Used [PCIe ID: 0000:8d:00.0]
    Device  6 [nvidia-smi  5]:          Tesla C1060 (CC 1.3):   106 of  4095 MiB Used [PCIe ID: 0000:87:00.0]
    Device  7 [nvidia-smi  4]:          Tesla C2075 (CC 2.0):   115 of  6143 MiB Used [PCIe ID: 0000:86:00.0]

