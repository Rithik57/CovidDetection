import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
import tensorflow as tf
print(tf.test.gpu_device_name());
print("CPU count : ",os.cpu_count());