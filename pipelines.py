import numpy as np
from multiprocessing import Pool
import tensorflow as tf
import main as m

# Pipeline strategy interface
class PipelineStrategy(object):
    ''' Simulated abstract base class for pipeline strategy '''

    def preprocess(self, features, labels, reshape):
        raise NotImplementedError

    def split_dataset(self, mode):
        raise NotImplementedError
        
    def get_next_batch(self, mode):
        raise NotImplementedError  
    
    def feed_dict(self, next_batch, mode):
        raise NotImplementedError
    
    def initializers(self, iterator):
        raise NotImplementedError

    def transform(dataset):
        raise NotImplementedError

    def augment(feature, label):
        raise NotImplementedError
        
    def shuffle(dataset):
        raise NotImplementedError

# Pipeline strategy 1
class FeedDictPipeline(PipelineStrategy):
    def __init__(self, features, labels, params):
        self.EPOCHS_COMPLETED = 0
        self.INDEX_IN_EPOCH = 0
        print(params['NUM_EXAMPLES'])
        self.features, self.labels = self.shuffle(features, labels, params['NUM_EXAMPLES'])
        self.params = params

    def get_next_batch(self, mode=None):
        '''Placeholders only'''
        replace_index = lambda tuple_: (None,) + tuple_[1:] # to turn NHWC to ?HWC
        return tuple((tf.placeholder(self.features.dtype, replace_index(self.features.shape)),
        tf.placeholder(self.labels.dtype, replace_index(self.labels.shape))
        ))
    
    def feed_dict(self, next_batch, mode):
        '''Returns a dictionary of next batch'''
        features, labels, buff = self.split_dataset(mode)
        start = self.INDEX_IN_EPOCH
        self.INDEX_IN_EPOCH += self.params['BATCH_SIZE']
        if self.INDEX_IN_EPOCH > self.params['TRAIN_SIZE']: # fix to include validation
            self.EPOCHS_COMPLETED += 1
            self.shuffle(features, labels, buff)
            start = 0
            self.INDEX_IN_EPOCH = self.params['BATCH_SIZE']
            assert self.params['BATCH_SIZE'] <= self.params['TRAIN_SIZE']
        end = self.INDEX_IN_EPOCH
        return {next_batch: (self.features[start:end], self.labels[start:end])}

    def split_dataset(self, mode):
        if mode == 'training':
            return (self.features[:self.params['TRAIN_SIZE']], self.labels[:self.params['TRAIN_SIZE']], 
            self.params['TRAIN_SIZE'])
        elif mode == 'validation':
            return (self.features[self.params['TRAIN_SIZE']:], self.labels[self.params['TRAIN_SIZE']:],
            self.params['NUM_EXAMPLES'] - self.params['TRAIN_SIZE'])
        else:
            raise ArgumentError
    
    def initializers(iterator):
        return tf.global_variables_initializer()
    
    def transform(self, dataset):
        p = Pool(self.params['NUM_CORES'])
        return p.map(self.augment, zip(*dataset))

    def augment(self, example):
        return example
        
    def shuffle(self, features, labels, buff):
        perm = np.arange(buff)
        np.random.shuffle(perm)
        return features[perm], labels[perm]

# pipeline strategy 2
class DataAPIPipeline(PipelineStrategy):
    def __init__(self, features, labels, params):
        self.params = params
        self.full_dataset = tf.data.Dataset.from_tensor_slices((features, labels)).shuffle(self.params['NUM_EXAMPLES'])
        self.iterator = None
    
    def get_next_batch(self, mode):
        dataset, buff = self.split_dataset(mode)
        dataset = dataset.cache().shuffle(buff).repeat().batch(self.params['BATCH_SIZE']).prefetch(1)
        # dataset = dataset.shuffle(buff).repeat().batch(self.params['BATCH_SIZE'])
        self.iterator = dataset.make_initializable_iterator()
        return self.iterator.get_next()
    
    def feed_dict(self, next_batch=None, mode=None):
        return None 

    def split_dataset(self, mode):
        if mode == 'training':
            return (self.full_dataset.take(self.params['TRAIN_SIZE']), self.params['TRAIN_SIZE'])
        elif mode == 'validation':
            return (self.full_dataset.skip(self.params['TRAIN_SIZE']), 
            self.params['NUM_EXAMPLES'] - self.params['TRAIN_SIZE'])
        else:
            raise ArgumentError

    def initializers(self):
        return [tf.global_variables_initializer(), self.iterator.initializer] 

    def transform(self, dataset):
        return dataset.map(lambda feature, label: self.augment(feature, label))

    def augment(self, feature, label):
        #feature = tf.image.random_hue(feature, 0.5)
        pass
   
    def shuffle(self, dataset):
        pass