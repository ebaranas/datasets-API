# IN BENCH-SCRIPT
# basic but cleaned up version for benchmarking tf.data API
# datasets: toy, bfimage v42 in numpy arrays
# models: fully-connected, CNN LowResClassifier
# training and validation only
# includes MNIST data

# environment
# pip install python-mnist
# pip install toml

# Bunch of imports
import tensorflow as tf
import numpy as np
import os
import sys
from mnist import MNIST
import math
import benchmark as bm

def main():
    # GPU utilization
    CONFIG_PATH = "/project/datasets-API/benchmark-package/config/mnist.toml"
    NUM_RUNS = 1
    PATH_TO_CHROME_TRACES = "/project/datasets-API/benchmark-package/chrome-traces/test.json"
    
    benchmark = bm.Benchmark('data_API', 'LowResFrameClassifier', CONFIG_PATH)
    tot_acc, tot_time = 0, 0
    for i in range(NUM_RUNS):
        tot_time += benchmark.run('training', 500, profile=False)[0]
        # tot_acc += benchmark.run('validation', 1000, profile=False)[1]
    print("RESULTS: Train time=", tot_time/NUM_RUNS, ", Validation accuracy=", tot_acc/NUM_RUNS*100)
    
    # # EFFECT OF PREFETCH AND CACHE
    # CONFIG_PATH = "/project/datasets-API/benchmark-package/config/bfimage.toml"
    # NUM_RUNS = 10
    # PATH_TO_CHROME_TRACES = "/project/datasets-API/benchmark-package/chrome-traces/test.json"
    
    # benchmark = bm.Benchmark('data_API', 'FullyConnected', CONFIG_PATH)
    # tot_acc, tot_time = 0, 0
    # for i in range(NUM_RUNS):
    #     tot_time += benchmark.run('training', 2000, profile=False)[0]
    #     tot_acc += benchmark.run('validation', 1000, profile=False)[1]
    # print("RESULTS: Train time=", tot_time/NUM_RUNS, ", Validation accuracy=", tot_acc/NUM_RUNS*100)
    
    
    # EFFECT OF DTYPES
    # CONFIG_PATH = "/project/datasets-API/benchmark-package/config/bfimage.toml"
    # NUM_RUNS = 10
    # PATH_TO_CHROME_TRACES = "/project/datasets-API/benchmark-package/chrome-traces/test.json"
    
    # benchmark = bm.Benchmark('data_API', 'FullyConnected', CONFIG_PATH)
    # tot_acc, tot_time = 0, 0
    # for i in range(NUM_RUNS):
    #     tot_time += benchmark.run('training', 2000, profile=False)[0]
    #     tot_acc += benchmark.run('validation', 1000, profile=False)[1]
    # print("RESULTS: Train time=", tot_time/NUM_RUNS, ", Validation accuracy=", tot_acc/NUM_RUNS*100)
    

    
    # FIXED PARAMS: BFIMAGE + LOWRES RUN
    # CONFIG_PATH = "/project/datasets-API/benchmark-package/config/bfimage.toml"
    # NUM_RUNS = 5
    # PATH_TO_CHROME_TRACES = "/project/datasets-API/benchmark-package/chrome-traces/test.json"
    
    # benchmark = bm.Benchmark('data_API', 'LowResFrameClassifier', CONFIG_PATH)
    # tot_acc, tot_time = 0, 0
    # for i in range(NUM_RUNS):
    #     tot_time += benchmark.run('training', 2000, profile=False)[0]
    #     tot_acc += benchmark.run('validation', 1000, profile=False)[1]
    # print("RESULTS: Train time=", tot_time/NUM_RUNS, ", Validation accuracy=", tot_acc/NUM_RUNS*100)
    
    
    # FIXED PARAMS: MNIST + LOWRES RUN
    # CONFIG_PATH = "/project/datasets-API/benchmark-package/config/mnist.toml"
    # NUM_RUNS = 10
    # PATH_TO_CHROME_TRACES = "/project/datasets-API/benchmark-package/chrome-traces/test.json"
    
    # benchmark = bm.Benchmark('data_API', 'LowResFrameClassifier', CONFIG_PATH)
    # tot_acc, tot_time = 0, 0
    # for i in range(NUM_RUNS):
    #     tot_time += benchmark.run('training', 2000, profile=False)[0]
    #     tot_acc += benchmark.run('validation', 1000, profile=False)[1]
    # print("RESULTS: Train time=", tot_time/NUM_RUNS, ", Validation accuracy=", tot_acc/NUM_RUNS*100)
    
    # FIXED PARAMS: BFIMAGE + FC RUN
    # CONFIG_PATH = "/project/datasets-API/benchmark-package/config/bfimage.toml"
    # NUM_RUNS = 10
    # PATH_TO_CHROME_TRACES = "/project/datasets-API/benchmark-package/chrome-traces/test.json"
    
    # benchmark = bm.Benchmark('data_API', 'FullyConnected', CONFIG_PATH)
    # tot_acc, tot_time = 0, 0
    # for i in range(NUM_RUNS):
    #     tot_time += benchmark.run('training', 2000, profile=False)[0]
    #     tot_acc += benchmark.run('validation', 1000, profile=False)[1]
    # print("RESULTS: Train time=", tot_time/NUM_RUNS, ", Validation accuracy=", tot_acc/NUM_RUNS*100)
    
    
    # FIXED PARAMS: MNIST + FC  RUN
    # CONFIG_PATH = "/project/datasets-API/benchmark-package/config/mnist.toml"
    # NUM_RUNS = 10
    # PATH_TO_CHROME_TRACES = "/project/datasets-API/benchmark-package/chrome-traces/test.json"
    
    # benchmark = bm.Benchmark('data_API', 'FullyConnected', CONFIG_PATH)
    # tot_acc, tot_time = 0, 0
    # for i in range(NUM_RUNS):
    #     tot_time += benchmark.run('training', 10000, profile=False)[0]
    #     tot_acc += benchmark.run('validation', 5000, profile=False)[1]
    # print("RESULTS: Train time=", tot_time/NUM_RUNS, ", Validation accuracy=", tot_acc/NUM_RUNS*100)

if __name__ == "__main__":
    main()