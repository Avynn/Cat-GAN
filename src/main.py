import tensorflow as tf
import imagePipeline

#get the training images paths
trainCats = tf.constant([("../resources/training_set/cats/cat.%d.jpg" % i) for i in range(1,4000)])
trainDogs = tf.constant([("../resources/training_set/dogs/dog.%d.jpg" % i) for i in range(1,4000)])
trainingSet = tf.concat([trainCats, trainDogs], 0)

#create batches
inputImages, inputLbaels = imagePipeline.inputPipeline(trainingSet, 800)

with tf.Session() as sess:
        #init variables in session
        tf.global_variables_initializer().run()

        #init coordinator and start queing dem bois up!
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        

        #deinit coord
        coord.request_stop()
        coord.join(threads)