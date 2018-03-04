from PIL import Image
import random
import numpy as np
import tensorflow as tf
import os
import io
import sys

class FileNameQueue:
    def __init__(self, folderPath, numEpochs):
        # print("ping!")
        self.contents = []

        for i in range(numEpochs):
            self.enqueuePaths(folderPath)


    def enqueuePaths(self, folderPath):
        i = 0
        for (dirPath, dirNames, filenames) in os.walk(folderPath):
            if(len(dirNames) == 0):
                for fileName in filenames:
                    self.contents.append(dirPath + "/" + fileName)
        i += 1

    def dequeue(self):
        return self.contents.pop()

    def isEmpty(self):
        if(len(self.contents) == 0):
            return True
        else:
            return False

    def dump(self):
        return self.contents

def readIMG(pathQueue):
    reader = tf.WholeFileReader()
    key, value = reader.read(pathQueue)
    decodedExample = tf.image.decode_jpeg(value, channels=3)
    exampleFull = tf.image.resize_images(decodedExample, [64,64])
    example = tf.reshape(exampleFull, [4096, 3])
    return example, key


def getLabel(paths):
    shape = paths[0].shape

    out = np.zeros((shape[0], 2))
    for i in range(shape[0]):
        if("cat" in paths[0][i]):
            out[i, 0] = 1
        else:
            out[i, 1] = 1
    return tf.convert_to_tensor(out)

def inputPipeline(folderPath, batchSize, numEpochs):
        minAfterDequeue = 100
        numThreads = 3
        filenames = FileNameQueue(folderPath, numEpochs)
        queue = tf.train.string_input_producer(filenames.dump())
        example, label = readIMG(queue)
        capacity = minAfterDequeue + numThreads + 3 * batchSize
        exampleBatch, labelBatch = tf.train.shuffle_batch([example, label],
                                                        batch_size=batchSize,
                                                        num_threads=numThreads,
                                                        capacity=capacity,
                                                        min_after_dequeue=minAfterDequeue)
        return exampleBatch, labelBatch

