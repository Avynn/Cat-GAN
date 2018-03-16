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

    Wfc1 = weightVariable([100, 4096, 200])
    Bfc1 = biasVariable([100, 3, 200])
    tf.summary.histogram("layer1 weights", Wfc1)
    Hfc1 = tf.nn.relu(tf.matmul(xTransposed, Wfc1) + Bfc1)

    Wfc2 = weightVariable([100, 200, 1])
    Bfc2 = biasVariable([100, 3, 1])
    tf.summary.histogram("layer2 weights", Wfc2)
    Hfc2 = tf.nn.relu(tf.matmul(Hfc1, Wfc2) + Bfc2)

    in3 = tf.transpose(Hfc2, perm=[0,2,1])
    Wfc3 = weightVariable([100, 3,2])
    Bfc3 = biasVariable([100, 1,2])
    tf.summary.histogram("layer3 weights", Wfc3)
    logits = tf.nn.softmax(tf.matmul(in3, Wfc3) + Bfc3)

    logits = tf.reshape(logits, [100,2])

    return logits, labels
