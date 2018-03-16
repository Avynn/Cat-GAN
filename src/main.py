import tensorflow as tf
from tensorflow.python import debug as tf_debug
import imagePipeline
import model
import backProp

#create batches
print("ping!")
inputImages, inputLabels = imagePipeline.inputPipeline("../resources", 100, 1)
logits, labels = model.model(inputImages, inputLabels)
accuracy = backProp.evalLogits(logits, labels)

with tf.Session() as sess:

        sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        #init variables in session
        # tf.global_variables_initializer().run()
        # tf.local_variables_initializer().run()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        #init coordinator and start queing dem bois up!
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)



        for i in range(30):  
                print(sess.run(accuracy))

        #deinit coord
        coord.request_stop()
        coord.join(threads)
