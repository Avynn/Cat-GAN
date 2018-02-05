import tensorflow as tf
import imagePipeline

#get the training images paths
trainCats = tf.constant([("../resources/training_set/cats/cat.%d.jpg" % i) for i in range(1,4000)], dtype=tf.string)
trainDogs = tf.constant([("../resources/training_set/dogs/dog.%d.jpg" % i) for i in range(1,4000)], dtype=tf.string)
trainingSet = tf.concat([trainCats, trainDogs], 0)

#create batches
inputImages, inputLabels = imagePipeline.inputPipeline(trainingSet, 100)

with tf.Session() as sess:
        #init variables in session
        tf.global_variables_initializer().run()

        #init coordinator and start queing dem bois up!
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run([inputImages])
        sess.run([inputLabels])

        #deinit coord
        coord.request_stop()
        coord.join(threads)
