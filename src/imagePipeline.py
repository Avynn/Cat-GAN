import random
import tensorflow as tf
import os
import sys

def readExamples(filenameQue):
        reader = tf.TFRecordReader()
        _, serializedOutput = reader.read(filenameQue)

        print("ping!")

        features = {
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'rawImage': tf.FixedLenFeature([], tf.string)
        }

        example = tf.parse_single_example(
            serializedOutput,
            features,
        )

        image = tf.decode_raw(example['rawImage'], tf.uint8)

        image = tf.reshape(image, [64,64,3])

        label = tf.cast(example['label'], tf.int32)

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


# def parseRecords(exapmlePrototype):
#     features = {
#         'label': tf.FixedLenFeature((), tf.int64, default_value=0),
#         'rawImage': tf.FixedLenFeature((), tf.string, default_value="")
#     }
#     parsedFeatures = tf.parse_single_example(exapmlePrototype, features)
#     image = tf.decode_raw(parsedFeatures['rawImage'], tf.uint8)
#     return image, parsedFeatures['label']
