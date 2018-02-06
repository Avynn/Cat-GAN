import random
import tensorflow as tf
import os
import sys

# trainCats = tf.constant([("../resources/training_set/cats/cat.%d.jpg" % i) for i in range(1,4000)], dtype=tf.string)
# trainDogs = tf.constant([("../resources/training_set/dogs/dog.%d.jpg" % i) for i in range(1,4000)], dtype=tf.string)
# trainingSet = tf.concat([trainCats, trainDogs], 0)


def readJPEG(filenameQue):
        label = tf.Variable([0], dtype=tf.int8)
        reader = tf.WholeFileReader()
        key, recordString = reader.read(filenameQue)
        examplePreProcess = tf.image.decode_jpeg(recordString)
        exampleShape = tf.shape(examplePreProcess)
        randX = random.randint(0, (exampleShape[0] - 32))
        randY = random.randint(0, (exampleShape[1] - 32))

        example = tf.reshape(tf.strided_slice(examplePreProcess, [randX,randY,0], [randX + 32, randY + 32, 3]), [32, 32, 3])
        return example, key

def inputPipeline(filenames, batchSize):
        minAfterDequeue = 10000
        numThreads = 3
        filenameQueue = tf.train.string_input_producer(filenames)
        example, label = readJPEG(filenameQueue)
        capacity = minAfterDequeue + (numThreads + 3) * batchSize
        exampleBatch, labelBatch = tf.train.shuffle_batch([example, label],
                                                        batch_size=batchSize,
                                                        num_threads=numThreads,
                                                        capacity=capacity,
                                                        min_after_dequeue=minAfterDequeue)
        return exampleBatch, labelBatch
