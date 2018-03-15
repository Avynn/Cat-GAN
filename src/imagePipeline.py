from PIL import Image
import random
import numpy as np
import tensorflow as tf
import os
import io
import sys

# class FileNameQueue:
#     def __init__(self, folderPaths):
#         # print("ping!")
#         self.contents = []
#
#         self.enqueuePaths(folderPaths)
#
#
#     def enqueuePaths(self, folderPath):
#         i = 0
#         for (dirPath, dirNames, filenames) in os.walk(folderPath):
#             if(len(dirNames) == 0):
#                 for fileName in filenames:
#                     path = dirPath + "/" + fileName
#                     self.contents.append(path)
#         i += 1
#
#     def dequeue(self):
#         return self.contents.pop()
#
#     def isEmpty(self):
#         if(len(self.contents) == 0):
#             return True
#         else:
#             return False
#
#     def dump(self):
#         return self.contents

def getPaths(folderPath):
    contents = []
    for (dirPath, dirNames, filenames) in os.walk(folderPath):
        if(len(dirNames) == 0):
            for fileName in filenames:
                path = dirPath + "/" + fileName
                contents.append(path)
    return contents

def readIMG(pathQueue):
    reader = tf.WholeFileReader()
    key, value = reader.read(pathQueue)
    decodedExample = tf.image.decode_jpeg(value, channels=3)
    exampleFull = tf.image.resize_images(decodedExample, [64,64])
    example = tf.reshape(exampleFull, [4096, 3])
    label = tf.py_func(getLabel, [key], tf.float32)
    label.set_shape([2])
    return example, label

def getLabel(path):
    arrToReturn = np.zeros((2))
    if(b"cat" in path):
        arrToReturn[0] = 1
        return arrToReturn
    else:
        arrToReturn[1] = 1
        return arrToReturn


def inputPipeline(folderPath, batchSize, numEpochs):
        minAfterDequeue = 100
        numThreads = 3
        filenames = tf.constant(getPaths(folderPath), dtype=tf.string)
        queue = tf.train.string_input_producer(filenames, num_epochs=numEpochs, capacity=10500)
        example, label = readIMG(queue)
        capacity = minAfterDequeue + numThreads + 3 * batchSize
        exampleBatch, labelBatch = tf.train.shuffle_batch([example, label],
                                                        batch_size=batchSize,
                                                        num_threads=numThreads,
                                                        capacity=capacity,
                                                        min_after_dequeue=minAfterDequeue)
        return exampleBatch, labelBatch
