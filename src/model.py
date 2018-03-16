import tensorflow as tf
import imagePipeline

def weightVariable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def biasVariable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def model(flatInputImages, labels):
    xDrop = tf.nn.dropout(flatInputImages, keep_prob=1.0)

    xTransposed = tf.transpose(xDrop, perm=[0,2,1])

    Wfc1 = weightVariable([100, 4096, 8192])
    Bfc1 = biasVariable([100, 3, 8192])
    Hfc1 = tf.nn.relu(tf.matmul(xTransposed, Wfc1) + Bfc1)

    Wfc2 = weightVariable([100, 8192, 75])
    Bfc2 = biasVariable([100, 3, 75])
    Hfc2 = tf.nn.relu(tf.matmul(Hfc1, Wfc2) + Bfc2)

    Wfc3 = weightVariable([100, 75, 5])
    Bfc3 = biasVariable([100,3,5])
    Hfc3 = tf.nn.relu(tf.matmul(Hfc2, Wfc3) + Bfc3)

    # in4 = tf.transpose(Hfc3, perm=[0,2,1])
    Wfc4 = weightVariable([100,5,2])
    Bfc4 = biasVariable([100,3,2])
    logits = tf.nn.softmax(tf.matmul(Hfc3, Wfc4) + Bfc4)

    averageActivation = (tf.reduce_sum(tf.transpose(logits, perm=[0,2,1]), axis=2) / 3)

    return averageActivation, labels
