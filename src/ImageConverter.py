from PIL import Image
import numpy as np
import tensorflow as tf
import argparse
import os
import glob
import sys
import io

SHAPEX = 64
SHAPEY = 64

def int64Feature(val):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))

def bytesFeature(val):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))

def convertImages(folderPath, name):
    knownDirectories = []
    outFile = "../resources/" + name + ".tfrecords"
    print("writing %s" % name)
    writer = tf.python_io.TFRecordWriter(outFile)
    i = 0
    for (dirPath, dirNames, fileNames) in os.walk(folderPath):
        if (i == 0):
            for j in range(len(dirNames)):
                knownDirectories.append(dirNames[j])
        else:
            print(knownDirectories)
            print(i)
            for fileName in fileNames:
                label = 0
                path = folderPath + '/' + knownDirectories[i - 1] + "/" + fileName
                image = Image.open(path)
                image = image.resize((SHAPEX,SHAPEY))
                imageToBytes = io.BytesIO()
                image.save(imageToBytes, format="png")
                if("cat" in fileName):
                    label += 1
                example = tf.train.Example(features = tf.train.Features(feature = {
                    'height': int64Feature(SHAPEY),
                    'width': int64Feature(SHAPEX),
                    'depth': int64Feature(3),
                    'label': int64Feature(label),
                    'rawImage': bytesFeature(imageToBytes)
                }))
                writer.write(example.serializeToString())

        i += 1
    writer.close()

def main(argv):
    convertImages('../resources/training_set', "training")
    convertImages('../resources/test_set', 'testing')

tf.app.run(main=main, argv=[sys.argv[0]])
