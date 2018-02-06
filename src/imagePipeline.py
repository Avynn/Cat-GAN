import random
import tensorflow as tf
import os
import sys

SHAPE = 64 * 64 * 3


def readExamples(filenameQue):
        reader = tf.TFRecordReader()
        _, serializedOutput = reader.read(filenameQue)
        print(type(serializedOutput))
        example = tf.train.Example()
        example.ParseFromString(serializedOutput)
        label = int(example.features.feature['label'].int64_list.value[0])
        imgBytes = example.features.feature['rawImage'].bytes_list.value[0]
        image = tf.decode_raw(imgBytes)
        image.set_shape(SHAPE)

        return image, label

def inputPipeline(filename, batchSize, numEpochs):
        minAfterDequeue = 1000
        numThreads = 3
        filenameQueue = tf.train.string_input_producer([filename], num_epochs=numEpochs)
        example, label = readExamples(filenameQueue)
        capacity = minAfterDequeue + numThreads + 3 * batchSize
        exampleBatch, labelBatch = tf.train.shuffle_batch([example, label],
                                                        batch_size=batchSize,
                                                        num_threads=numThreads,
                                                        capacity=capacity,
                                                        min_after_dequeue=minAfterDequeue)
        return exampleBatch, labelBatch
