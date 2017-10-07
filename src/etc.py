from keras.applications import VGG16
from keras import backend as K
import tensorflow as tf

sess = tf.Session()
K.set_session(sess)

base_model = VGG16(include_top=False, weights='imagenet')
base_model.summary()

layer = base_model.get_layer('block1_conv1')
orig_weights = layer.get_weights()

sess.run(tf.global_variables_initializer())

new_weights = layer.get_weights()

#print(orig_weights[0] - new_weights[0])

var = tf.Variable(0., name='var')
print(var.name)

need_init_names = sess.run(tf.report_uninitialized_variables())
print(need_init_names)
#need_init = [v.name for v in tf.global_variables()] # if var.name in need_init_names]
need_init = [v.name for v in tf.global_variables() \
        if not(sess.run(tf.is_variable_initialized(v)))]
print(need_init)
