import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import numpy as np
import time
from squeezenet inport *

from PIL import Image

from argparse import ArgumentParser

%matplotlib inline


model = load_sqz_model("pretrained-model/sqz_full.mat")

def content_loss(content_weight, content_current, content_original):

    shapes = tf.shape(content_current)
    
    F_l = tf.reshape(content_current, [shapes[1], shapes[2]*shapes[3]])
    P_l = tf.reshape(content_original,[shapes[1], shapes[2]*shapes[3]])
    
    loss = content_weight * (tf.reduce_sum((F_l - P_l)*(F_l - P_l)))
    
    return loss

def gram_matrix(features, normalize=True):
 
    
    shapes = tf.shape(features)
    
    F_l = tf.reshape(features, shape=[shapes[1]*shapes[2],shapes[3]])
    
    gram = tf.matmul(tf.transpose(F_l),F_l)
    
    if normalize == True:
        gram /= tf.cast(shapes[1]*shapes[2]*shapes[3],tf.float32)
    
    return gram

def style_loss(feats, style_layers, style_targets, style_weights):

    style_loss = tf.constant(0.0)
    
    for i in range(len(style_layers)):
        current_im_gram = gram_matrix(feats[style_layers[i]])
        tmp = style_weights[i] * tf.reduce_sum((current_im_gram - style_targets[i])*(current_im_gram - style_targets[i]))
        style_loss += tmp
    
    return style_loss

def total_loss(content_loss, style_loss, alpha = 10, beta = 40):
 
    Loss = alpha * content_loss + beta * style_loss
    
    return Loss


content_image = scipy.misc.imread("images/own_content.jpg")
print(content_image.size)
content_image = reshape_and_normalize_image(content_image)


style_image = scipy.misc.imread("images/own_style.jpg")
print(style_image.size)
style_image = reshape_and_normalize_image(style_image)


generated_image = generate_noise_image(content_image)

model = load_sqz_model("pretrained-model/sqz_full.mat")


sess.run(model['input'].assign(content_image))

out = model['conv4_2']

a_C = sess.run(out)


a_G = out
m, n_H, n_W, n_C = a_G.get_shape().as_list()
content_loss = tf.reduce_sum(tf.squared_difference(tf.reshape(tf.transpose(a_C,[0,3,1,2]),[m,n_C,-1]),tf.reshape(tf.transpose(a_G,[0,3,1,2]),[m,n_C,-1])))/(n_H*n_W*n_C*4)
#content = content_loss(content_weight, content_current, content_original)

sess.run(model['input'].assign(style_image))

 def compute_style_cost(model, STYLE_LAYERS):

    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        out = model[layer_name]

        a_S = sess.run(out)

       
        a_G = out
        
        m, n_H, n_W, n_C = a_G.get_shape().as_list()
		a_S = tf.reshape(tf.transpose(a_S,[0,3,1,2]),[m,n_C,n_H*n_W])
    	a_G = tf.reshape(tf.transpose(a_G,[0,3,1,2]),[m,n_C,n_H*n_W])
    	GA = tf.matmul(A,A,transpose_b=True)
    	GS = tf.matmul(a_S,a_S,transpose_b=True)
    	GG = tf.matmul(a_G,a_G,transpose_b=True)
    	tmp = tf.reduce_sum(tf.squared_difference(GS,GG))/(4*(n_C*n_C)*((n_H*n_W)*(n_H*n_W)))

        J_style += coeff * tmp

    return J_style




style = style_loss(model, STYLE_LAYERS)

loss = total_loss(content,style,10,40)

optimizer = tf.train.AdamOptimizer(2.0)

train_step = optimizer.minimize(loss)

def model_nn(sess, input_image):

    sess.run(tf.global_variables_initializer())

    sess.run(model['input'].assign(input_image))
    
    for i in range(150):
    

        sess.run(train_step)

        generated_image = sess.run(model['input'])

        if i%20 == 0:

            
            save_image("output/" + str(i) + ".png", generated_image)
    
    save_image('output/generated_image.jpg', generated_image)
    
    return generated_image

 model_nn(sess, generated_image)
