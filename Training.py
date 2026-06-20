from Preprocessing import datasetTrain, datasetTest, datasetVal, VOCABSIZE
import tensorflow as tf

EMBEDDINGDIMS = 128
HIDDENSIZE = 256

class Encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(VOCABSIZE, EMBEDDINGDIMS)
        self.lstm = tf.keras.layers.LSTM(HIDDENSIZE, return_sequences=True)
    
    def call(self, text):
        text = self.embedding(text)
        text = self.lstm(text)
        return text

batchSize = 16
textLength = 64
encoder = Encoder()
t = tf.random.uniform(shape=(batchSize, textLength), minval=0, maxval=VOCABSIZE, dtype=tf.int32)
print(encoder(t).shape)