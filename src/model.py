from keras.applications import VGG16
from keras import backend as K
from utils import *
import numpy as np
import tensorflow as tf
from tqdm import trange 

#Model arguments
np.random.seed(416)
tf.set_random_seed(256)

learn_rate = 1. 
n_iter = 1000

content_layer_name = 'block2_conv2'
content_weight = 1.

sess = tf.Session()
K.set_session(sess)


#Load images
content_image_path = '../img/portrait.jpg'
content_image = load_prep_image(content_image_path)
whitenoise_image = np.random.normal(size=content_image.shape, scale=16)


#Build model
X = tf.Variable(whitenoise_image, name='X')
base_model = VGG16(input_tensor=X, include_top=False, weights='imagenet')
#base_model.summary()


#Build content loss
content_layer = base_model.get_layer(content_layer_name).output
content_target_np = sess.run(content_layer, feed_dict={X: content_image})
content_target = tf.Variable(content_target_np, name='content_target')
print(type(content_target_np), content_target_np.shape)
content_loss = tf.reduce_mean(tf.square(content_layer - content_target))


#Build style loss


#Model loss
loss = content_loss * content_weight 


#Optimizing
adam_op = tf.train.AdamOptimizer(learn_rate).minimize(loss, var_list=[X])
need_init = [v for v in tf.global_variables() if not(sess.run(tf.is_variable_initialized(v)))]
sess.run(tf.variables_initializer(need_init))
t = trange(n_iter)
for i in t:
    sess.run(adam_op)
    current_loss = sess.run(loss)
    t.set_description('Iter %d Loss: %f' % (i, current_loss))


#Save image
syn_image = sess.run(X)
save_deprep_image('../img/syn_portrait.jpg', syn_image)
    


