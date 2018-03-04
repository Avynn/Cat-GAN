import tensorflow as tf
import imagePipeline
import model

#create batches
inputImages, inputLabels = imagePipeline.inputPipeline("../resources", 100, 1)
logits, labels = model.model(inputImages, inputLabels)


with tf.Session() as sess:
        #init variables in session
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        # sess.run(iterator.initializer)
        # print(iterator.output_types)
        #init coordinator and start queing dem bois up!
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # printList = []
        # sess.run([nextBatch])

        print(sess.run([logits]))

        # print(inputImages)

        #deinit coord
        coord.request_stop()
        coord.join(threads)
