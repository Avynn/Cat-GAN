import numpy as np
import tensorflow as tf
#get the training images paths
trainCats = tf.constant([("/Users/avynn/Documents/codingProjects/catGAN/resources/cats/cat.%d.jpg" % i) for i in range(4000)])
# trainDogs = tf.constant([("/resources/training_set/dogs/dog.%d.jpg" % i) for i in range(4000)])
# trainingSet = tf.add(trainCats, trainDogs)

#enque the image paths
fileStringQueue = tf.train.string_input_producer(trainCats, capacity=80000)

#create and read the images
imageReader = tf.WholeFileReader()
_, imageFile = imageReader.read(fileStringQueue)

#decode images
image = tf.image.decode_jpeg(imageFile)

with tf.Session() as sess:
        #init variables in session
        tf.global_variables_initializer().run()

        #init coordinator and start queing dem bois up!
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        #get that image tensor!
        imageTensor = sess.run([image])

        #deinit coord
        coord.request_stop()
        coord.join(threads)