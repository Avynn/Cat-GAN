import tensorflow as tf
# from tensorflow.python import debug as tf_debug
import imagePipeline
import model
import backProp

#create batches
inputImages, inputLabels = imagePipeline.inputPipeline("../resources", 100, 1)
logits, labels = model.model(inputImages, inputLabels)
accuracy, trainStep = backProp.evalLogits(logits, labels)
summary = tf.summary.merge_all()

writer = tf.summary.FileWriter("../tensorBoard/", sess.graph)

with tf.Session() as sess:

        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        #init variables in session
        # tf.global_variables_initializer().run()
        # tf.local_variables_initializer().run()
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
