import os
import time
import numpy as np
import tensorflow as tf
from models import Model
from pipelines import DataAPIPipeline
from pipelines import FeedDictPipeline
import toml
from tensorflow.python.client import timeline
from multiprocessing import Pool
import data as D

class Benchmark(object):
    def __init__(self, pipeline, model_name, config_path):
        params = load_params(config_path)
        print("Training '{model}' on '{data}' using '{pipeline}'".format(model=model_name,
            data=params['DATA_NAME'], pipeline=pipeline))
        raw_features, raw_labels = D.read_raw(params['DATA_NAME'], params['DATA_PATH'])
        
        params.update({'NUM_EXAMPLES': len(raw_features)})
        params.update({'TRAIN_SIZE': int(params['TRAIN_RATIO']*params['NUM_EXAMPLES'])})
        self.params = params
        self.model = Model(model_name, params)
        SUPPORTED_PIPELINES = {"feed_dict": FeedDictPipeline, "data_API": DataAPIPipeline}
        PipelineClass = SUPPORTED_PIPELINES.get(pipeline, 'KeyError')
        
        reshape = self.should_reshape(rank=len(raw_features.shape))
        dtype = eval(params['DTYPE'])
        features, labels = self.preprocess(raw_features, raw_labels, dtype, reshape)
        
        self.pipeline = PipelineClass(features, labels, params)
        print("PARAMS: reshape=", reshape, " from ", raw_features.shape, " dtype=", dtype, " batch size=", 
        params['BATCH_SIZE'], " train size=", params['TRAIN_SIZE'])

    
    def preprocess(self, features, labels, dtype, reshape):
        assert len(features) == len(labels)
        if not isinstance(features, dtype):
            features = dtype(features)
        if reshape is not False:
            features = features.reshape(reshape)
        labels = np.array(Pool(self.params['NUM_CORES']).map(self.encode_to_one_hot, labels))
        return np.asarray(features), np.asarray(labels)
    
    def encode_to_one_hot(self, label):
        one_hot_encoding = np.zeros(self.params['NUM_CLASSES'])
        one_hot_encoding[label] = 1
        return one_hot_encoding    
    
    def run(self, mode, iterations, profile):
        next_batch = self.pipeline.get_next_batch(mode)
        fetches = self.model.train(next_batch)
        if profile is not False:
            # graph_writer = tf.summary.FileWriter("pipeline", sess.graph)
            run_metadata = tf.RunMetadata()
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        else:
            run_metadata = None
            options = None
        
        with tf.Session() as sess:
            sess.run(self.pipeline.initializers(), options=options, run_metadata=run_metadata)
            avg_acc = 0
            print("START BENCHMARK")
            start = time.time()
            for i in range(iterations):
                loss, optimizer, accuracy = sess.run(fetches,
                feed_dict=self.pipeline.feed_dict(next_batch, mode),
                options=options, run_metadata=run_metadata)
                avg_acc += accuracy
                if i % int(iterations/10) == 0:
                    print("Epoch: {}, loss: {:.3f}, {} accuracy: {:.2f}%".format(i, loss, mode, accuracy * 100))
            
            print("Average {} accuracy over {} iterations is {:.2f}%".format(mode, iterations, (avg_acc / iterations) * 100))
            end = time.time()
            time_per_run = end - start
            avg_acc = avg_acc/iterations
            
            print("END BENCHMARK. TIME: ", time_per_run)
        
            self.generate_trace(run_metadata, profile)
            return time_per_run, avg_acc
    
    def generate_trace(self, run_metadata, profile):
        if profile is False:
            return
        else:
            # Create the Timeline object, and write it to a json file
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open(profile, 'w') as f:
                f.write(chrome_trace)
            return

    def should_reshape(self, rank):
        if rank not in self.model.get_required_dims():
            if rank == 4: # e.g. bfimage, FullyConnected
                return (-1, self.params['HEIGHT']*self.params['WIDTH']*self.params['CHANNELS'])
            elif rank == 2: # e.g. mnist, LowResFrameClassifier
                return (-1, self.params['HEIGHT'], self.params['WIDTH'], self.params['CHANNELS'])
            else:
                ValueError("Invalid data shape")
        else:
            return False

def load_params(config_path, param={}):
    '''
    Load parameters from file.
    '''
    params = {}
    if not os.path.isfile(config_path):
        raise KeyError
    else:
        with open(config_path, 'r', encoding='utf-8') as f:
            params.update(toml.load(f))
        return params