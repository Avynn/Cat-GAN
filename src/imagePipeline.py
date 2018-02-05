import tensorflow as tf

def readJPEG(filenameQue):
        label = tf.Variable([0], dtype=tf.int8)
        reader = tf.WholeFileReader()
        key, recordString = reader.read(filenameQue)
        examplePreProcess = tf.image.decode_jpeg(recordString)
        example = tf.reshape(tf.strided_slice(examplePreProcess, [0,0,0], [32,32,3]), [32, 32, 3])
        return example, key

def inputPipeline(filenames, batchSize):
        minAfterDequeue = 10000
        numThreads = 16
        filenameQueue = tf.train.string_input_producer(filenames)
        example, label = readJPEG(filenameQueue)
        capacity = minAfterDequeue + (numThreads + 3) * batchSize
        exampleBatch, labelBatch = tf.train.shuffle_batch([example, label], 
                                                        batch_size=batchSize,
                                                        num_threads=numThreads,
                                                        capacity=capacity, 
                                                        min_after_dequeue=minAfterDequeue)
        return exampleBatch, labelBatch 