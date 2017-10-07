from keras.applications import VGG19
from keras import backend as K
from utils import *
import numpy as np
import tensorflow as tf
from tqdm import trange 
import atexit

try:
#Model arguments
    np.random.seed(416)
    tf.set_random_seed(256)

    target_size = (224, 224)

    learn_rate = 0.1 
    n_iter = 5000

    content_layer_name = 'block4_conv4'
    content_weight = 1.
    style_layer_names = [('block1_conv1', 1.),
                         ('block2_conv1', 1.),
                         ('block3_conv1', 1.),
                         ('block4_conv1', 1.),
                         ('block5_conv1', 1.)]
    style_weight = 1.

    sess = tf.Session()
    K.set_session(sess)


#Load images
    content_image_path = '../img/portrait.jpg'
    content_image = load_prep_image(content_image_path, target_size)
    style_image_path = '../img/style3.jpg'
    style_image = load_prep_image(style_image_path, content_image.shape[1:3])
    whitenoise_image = np.random.normal(size=content_image.shape, scale=16)


#Build model
    X = tf.Variable(whitenoise_image, name='X')
    base_model = VGG19(input_tensor=X, include_top=False, weights='imagenet')
#base_model.summary()


#Build content loss
    content_layer = base_model.get_layer(content_layer_name).output
    content_target_arr = sess.run(content_layer, feed_dict={X: content_image})
    content_target = tf.Variable(content_target_arr, name='content_target')
#    print(type(content_target_arr), content_target_arr.shape)
    content_loss = tf.reduce_mean(tf.square(content_layer - content_target))


#Build style loss
    style_loss = 0.
    style_layer_losses = []
    for style_layer_name, layer_weight in style_layer_names:
        style_layer = base_model.get_layer(style_layer_name).output
        shp = tf.shape(style_layer)
        N = shp[3]
        M = shp[1] * shp[2]
        style_layer = tf.reshape(style_layer, (-1, M, N))
        trans_layer = tf.transpose(style_layer, (0, 2, 1))
        gram_matrix = tf.matmul(trans_layer, style_layer) / tf.cast(M, tf.float32)
        (style_target_arr, trans_target_arr, gram_target_arr) = \
                sess.run([style_layer, trans_layer, gram_matrix], feed_dict={X: style_image})
#        print(type(style_target_arr), style_target_arr.shape)
#        print(type(trans_target_arr), trans_target_arr.shape)
#        print(type(gram_target_arr), gram_target_arr.shape)
        gram_target = tf.Variable(gram_target_arr, name='gram_target')
        layer_loss = tf.reduce_sum(tf.square(gram_matrix - gram_target)) / tf.cast(4 * N * N, tf.float32)
        style_layer_losses.append(layer_loss)
        style_loss = style_loss + layer_loss * layer_weight


#Model loss
#loss = content_loss * content_weight 
    loss = style_loss * style_weight


#Optimizing
    adam_op = tf.train.AdamOptimizer(learn_rate).minimize(loss, var_list=[X])
    need_init = [v for v in tf.global_variables() if not(sess.run(tf.is_variable_initialized(v)))]
    sess.run(tf.variables_initializer(need_init))
    t = trange(n_iter)
    for i in t:
        sess.run(adam_op)
        all_losses = (loss,) + tuple(style_layer_losses)
        current_loss = sess.run(all_losses)
        t.set_description('Loss: %e %e %e %e %e %e' % (current_loss))

finally:
#Save image
    syn_image = sess.run(X)
    save_deprep_image('../img/syn_portrait.jpg', syn_image)
        

