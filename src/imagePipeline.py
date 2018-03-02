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
    example = tf.image.decode_jpeg(value)
    label = getLabel(key)
    return example, label


def getLabel(path):
    out = np.zeros((2))
    if("cat" in path):
        out[0] = 1
        return tf.convert_to_tensor(out)
    else:
        out[1] = 1
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


# def parseRecords(exapmlePrototype):
#     features = {
#         'label': tf.FixedLenFeature((), tf.int64, default_value=0),
#         'rawImage': tf.FixedLenFeature((), tf.string, default_value="")
#     }
#     parsedFeatures = tf.parse_single_example(exapmlePrototype, features)
#     image = tf.decode_raw(parsedFeatures['rawImage'], tf.uint8)
#     return image, parsedFeatures['label']
