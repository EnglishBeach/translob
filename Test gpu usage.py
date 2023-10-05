import tensorflow as tf
# tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print(tf.config.list_physical_devices('GPU'))
print(tf.test.gpu_device_name())