import tensorflow as tf

class Model(object):
    def __init__(self, model_name, params):
        self.NUM_CLASSES = params['NUM_CLASSES']
        SUPPORTED_MODELS= {"FullyConnected": (self.FullyConnected, [2]),
                    "LowResFrameClassifier": (self.LowResFrameClassifier, [4]),
                    "SimpleFullyConnected":(self.SimpleFullyConnected, [2])
        }
        self.model_function, self.required_dims = SUPPORTED_MODELS.get(model_name, 'KeyError')
        
    def get_model_function(self):
        return self.model_function
    
    def get_required_dims(self):
        # For now first element of list is default.
        return self.required_dims
    
    def train(self, next_batch):
        logit, label = self.model_function(next_batch[0]), next_batch[1]
        loss = self.get_loss(logit, label)
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        accuracy = self.get_accuracy(logit, label)
        return [loss, optimizer, accuracy]
    
    def get_loss(self, logit, label):
        softmax = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit)
        return tf.reduce_sum(softmax)

    def get_accuracy(self, logit, label):
        prediction = tf.argmax(logit, 1)
        equality = tf.equal(prediction, tf.argmax(label, 1))
        return tf.reduce_mean(tf.cast(equality, tf.float32))
        
    def FullyConnected(self, feature):
        #bn = tf.layers.batch_normalization(feature)
        fc1 = tf.layers.dense(feature, 50)
        fc2 = tf.layers.dense(fc1, 50)
        fc2 = tf.layers.dropout(fc2)
        flat = tf.layers.flatten(fc2) # added flatten layer 08/10/2018 to correct shape
        fc3 = tf.layers.dense(flat, self.NUM_CLASSES)
        return fc3
        
    def SimpleFullyConnected(self, feature):
        #bn = tf.layers.batch_normalization(feature)
        fc1 = tf.layers.dense(feature, 30)
        final = tf.layers.dense(fc1, self.NUM_CLASSES)
        return final
    
    def LowResFrameClassifier(self, feature):
        #bn = tf.layers.batch_normalization(feature)
        conv1 = tf.layers.conv2d(feature, 32, (3, 3), activation="relu")
        conv2 = tf.layers.conv2d(conv1, 32, (3, 3), activation="relu")
        maxpool2d_a = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(1, 1))
        dropout_a = tf.layers.dropout(maxpool2d_a, rate=0.25)
        
        conv3 = tf.layers.conv2d(dropout_a, 64, (3, 3), activation="relu")
        conv4 = tf.layers.conv2d(conv3, 64, (3, 3), activation="relu")
        maxpool2d_b = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(1, 1))
        dropout_b = tf.layers.dropout(maxpool2d_b, rate=0.25)
        
        flat = tf.layers.flatten(dropout_b)
        dense = tf.layers.dense(flat, 256, activation="relu")
        dropout_c = tf.layers.dropout(dense, rate=0.25)
        
        final = tf.layers.dense(dropout_c, self.NUM_CLASSES, activation="softmax")
        return final