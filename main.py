import numpy as np
import tensorflow as tf
#get the training images paths
trainCats = tf.constant([("resources/training_set/cats/cat.%d.jpg" % i) for i in range(4000)])
trainDogs = tf.constant([("resources/training_set/dogs/dog.%d.jpg" % i) for i in range(4000)])
trainingSet = tf.concat([trainCats, trainDogs], 0)


def readJPEG(filenameQue):
        label = tf.Variable([0], dtype=tf.int8)
        reader = tf.WholeFileReader()
        key, recordString = reader.read(filenameQue)
        example = tf.image.decode_jpeg(recordString)
        return example, key

def inputPipeline(filenames, batchSize):
        filenameQueue = tf.train.string_input_producer(filenames)
        example, label = readJPEG(filenameQueue)
        minAfterDequeue = 10000
        capacity = minAfterDequeue + 3 * batchSize
        exampleBatch, labelBatch = tf.train.shuffle_batch([example, label], 
                                                        batch_size=batchSize, 
                                                        capacity=capacity, 
                                                        min_after_dequeue=minAfterDequeue)
        return exampleBatch, labelBatch 


inputImages, inputLbaels = inputPipeline(trainingSet, 800)

with tf.Session() as sess:
        #init variables in session
        tf.global_variables_initializer().run()

        #init coordinator and start queing dem bois up!
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print(inputImages.getShape())

        #deinit coord
        coord.request_stop()
        coord.join(threads)