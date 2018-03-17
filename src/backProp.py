import tensorflow as tf
import model

def evalLogits(logits, labels):
    # crossEntropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits), 1))

    trainStep = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(crossEntropy)
    correctPrediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    tf.summary.scalar("Cross Entropy", crossEntropy)
    return tf.reduce_mean(tf.cast(correctPrediction, tf.float32)), trainStep
