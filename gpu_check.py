import tensorflow as tf

print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')

print("GPUs detected:", gpus)

print("Number of GPUs:", len(gpus))

