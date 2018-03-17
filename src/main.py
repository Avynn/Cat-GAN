import tensorflow as tf
# from tensorflow.python import debug as tf_debug
import imagePipeline
import model
import backProp

#create batches
inputImages, inputLabels = imagePipeline.inputPipeline("../resources", 100, 1)

#define cross entropy and accuracy through the model
crossEntropy, accuracy = model.model(inputImages, inputLabels)

#define the train step with a momentum optimizer
trainStep = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(crossEntropy)

#merge summaries
summary = tf.summary.merge_all()

with tf.Session() as sess:

        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        writer = tf.summary.FileWriter("../tensorBoard/", sess.graph)

        #init variables in session
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        #init coordinator and start queing dem bois up!
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)


        # print(sess.run([accuracy, loss]))

        for i in range(200):
                acc, summ, _ = sess.run([accuracy, summary, trainStep])
                print(acc)
                writer.add_summary(summ, i)

        #deinit coord
        coord.request_stop()
        coord.join(threads)
