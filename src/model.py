import tensorflow as tf

def activationSummary(x):
    """
    NAME: activation summary
    PRE: a tensor (x)
    POST: attatches listeners to tensor for 
    summary on tensorboard
    """
    tensorName = x.op.name
    tf.summary.histogram(tensorName, )