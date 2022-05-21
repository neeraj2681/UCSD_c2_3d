# -*- coding: utf-8 -*-
"""
Created on Sat May 21 18:54:51 2022

@author: neera
"""

#Defining an autoencoder
input_init_2d = layers.Input(shape = (64, 128, 3))
el_2d1 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_init_2d)
el_2d1_normalized = keras.layers.BatchNormalization()(el_2d1)
el_2d1_activated = keras.layers.Activation("selu")(el_2d1_normalized)
el_2d1_pooled = layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2))(el_2d1_activated)
el_2d2 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(el_2d1_pooled)
el_2d2_normalized = keras.layers.BatchNormalization()(el_2d2)
el_2d2_activated = keras.layers.Activation("selu")(el_2d2_normalized)
el_2d2_pooled = layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2))(el_2d2_activated)
el_2d3 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(el_2d2_pooled)
el_2d3_normalized = keras.layers.BatchNormalization()(el_2d3)
el_2d3_activated = keras.layers.Activation("selu")(el_2d3_normalized)
el_2d3_pooled = layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2))(el_2d3_activated)
#layers.Conv2D(512, (3, 3), padding = 'same'),
#keras.layers.BatchNormalization(),
#keras.layers.Activation("selu"),
#layers.MaxPool2D(pool_size = 2),'''
    
input_init_3d = layers.Input(shape=(8, 64, 128, 3))
el_3d1 = layers.Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same')(input_init_3d)
el_3d1_normalized = keras.layers.BatchNormalization()(el_3d1)
el_3d1_activated = keras.layers.Activation("selu")(el_3d1_normalized)
el_3d1_pooled = layers.MaxPool3D(pool_size = (2, 2, 2), strides = (2, 2, 2))(el_3d1_activated)
el_3d2 = layers.Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same')(el_3d1_pooled)
el_3d2_normalized = keras.layers.BatchNormalization()(el_3d2)
el_3d2_activated = keras.layers.Activation("selu")(el_3d2_normalized)
el_3d2_pooled = layers.MaxPool3D(pool_size = (2, 2, 2), strides = (2, 2, 2))(el_3d2_activated)
el_3d3 = layers.Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same')(el_3d2_pooled)
el_3d3_normalized = keras.layers.BatchNormalization()(el_3d3)
el_3d3_activated = keras.layers.Activation("selu")(el_3d3_normalized)
el_3d3_pooled = layers.MaxPool3D(pool_size = (2, 2, 2), strides = (2, 2, 2))(el_3d3_activated)
#layers.Conv2D(512, (3, 3), padding = 'same'),
#keras.layers.BatchNormalization(),
#keras.layers.Activation("selu"),
#layers.MaxPool2D(pool_size = 2),'''

d3_latenet_mod_input = layers.Reshape((8, 16, 128))(el_3d3_pooled)
concat = layers.concatenate([el_2d3_pooled, d3_latenet_mod_input], axis = 3)
d3_latent_mod_output = layers.Reshape((1, 8, 16, 256))(concat)

dl_2d1 = layers.Conv2DTranspose(128, kernel_size = (3, 3), padding = "same", strides = (2, 2))(concat)
dl_2d1_normalized = keras.layers.BatchNormalization()(dl_2d1)
dl_2d1_activated = keras.layers.Activation("selu")(dl_2d1_normalized)
dl_2d2 = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', strides = (2, 2))(dl_2d1_activated)
dl_2d2_normalized = keras.layers.BatchNormalization()(dl_2d2)
dl_2d2_activated = keras.layers.Activation("selu")(dl_2d2_normalized)
dl_2d3 = layers.Conv2DTranspose(32, kernel_size = (3, 3), padding = "same", strides = (2, 2))(dl_2d2_activated)
dl_2d3_normalized = keras.layers.BatchNormalization()(dl_2d3)
dl_2d3_activated = keras.layers.Activation("selu")(dl_2d3_normalized)
dl_2d_final = layers.Conv2D(3, kernel_size = (3, 3), strides = 1, activation = "sigmoid", padding = "same")(dl_2d3_activated)
#layers.Reshape((158, 238, 3))

    
dl_3d1 = layers.Conv3DTranspose(128, kernel_size = (3, 3, 3), padding = "same", strides = (1, 2, 2))(d3_latent_mod_output)
dl_3d1_normalized = keras.layers.BatchNormalization()(dl_3d1)
dl_3d1_activated = keras.layers.Activation("selu")(dl_3d1_normalized)
dl_3d2 = layers.Conv3DTranspose(filters=64, kernel_size=(3, 3, 3), padding='same', strides = (1, 2, 2))(dl_3d1_activated)
dl_3d2_normalized = keras.layers.BatchNormalization()(dl_3d2)
dl_3d2_activated = keras.layers.Activation("selu")(dl_3d2_normalized)
dl_3d3 = layers.Conv3DTranspose(32, kernel_size = (3, 3, 3), padding = "same", strides = (1, 2, 2))(dl_3d2_activated)
dl_3d3_normalized = keras.layers.BatchNormalization()(dl_3d3)
dl_3d3_activated = keras.layers.Activation("selu")(dl_3d3_normalized)
dl_3d_final = layers.Conv3D(3, kernel_size = (3, 3, 3), strides = 1, activation = "sigmoid", padding = "same")(dl_3d3_activated)
#layers.Reshape((158, 238, 3))
    

model = tf.keras.Model(inputs = [input_init_3d, input_init_2d], outputs = [dl_3d_final, dl_2d_final])
model.compile(optimizer = 'adam', loss = losses.MeanSquaredError())