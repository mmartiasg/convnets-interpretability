#!/usr/bin/env python
# coding: utf-8

# # Part 2 of interpretability of a Convnet model
# 
# 
# In this case the idea now is to plot what the filters see when they activate at their highest level

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import math


# # load the 3 models we already've trained

# In[2]:


model1 = tf.keras.models.load_model("./checkpoints/model1/")

model2 = tf.keras.models.load_model("./checkpoints/model2/")

model3 = tf.keras.models.load_model("./checkpoints/model3/")


# # First step build a loss function
# 
# The idea is to maximize the activation

# In[3]:


def loss(activation):
    """
        initial_random_image: initial random image made to max out the activation of the filter.
        prediction: output of the filter's activation.
        Squared Error

        # Trying to maximize the activations average.
    """

    return tf.reduce_mean(activation)


# In[4]:


def preproces_img(image_output):

    filter_patern = image_output.copy()
    
    filter_patern -= filter_patern.mean()
    filter_patern /= filter_patern.std()
    filter_patern *= 64
    filter_patern += 128
    filter_patern = np.clip(filter_patern, 0, 255).astype("uint8")

    #Center image
    return filter_patern[25:-25, 25:-25, :]


# In[5]:


def get_convs_layers(model):
    outputs=[]
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D)):
            layer.trainable=False
            outputs.append(layer)
    return outputs


# In[6]:


def print_filters(model_name, layer_name, all_images, image_shape=(256, 256)):       
    margin = 5
    # all_images = [preproces_img(image) for image in generated_image_from_filters]

    # print(len(all_images))

    c = 4
    r = math.ceil(len(all_images)/c)

    # print(c, r)

    cropped_width = image_shape[0]-25*2
    cropped_height = image_shape[1]-25*2

    width = r * cropped_width + (r-1) * margin
    height = c * cropped_height + (c-1) * margin

    stitched_filters = np.zeros((width, height, 3))

    # print(width, height)

    #rows
    for i in tqdm(range(r)):
        for j in range(c):
            # print(i*c+j)
            image = all_images[i * c +j]

            row_start= (cropped_width + margin) * i
            row_end = (cropped_height + margin) * i + cropped_width
            column_start = (cropped_height + margin) * j
            column_end = (cropped_height + margin) * j + cropped_height

            # print("row_start, row_end, column_start, column_end")
            # print(row_start, row_end, column_start, column_end)

            stitched_filters[row_start: row_end, column_start: column_end, :] = image

    tf.keras.utils.save_img(f"./part2-filters/{model_name}_{layer_name}.png", stitched_filters)


# In[7]:


@tf.function
def optimize_step(feature_extractor_model, image, filter_index, lr):
    with tf.GradientTape() as tape:
        tape.watch(image)
        activation = feature_extractor_model(image)
        activation = activation[:, 2:-2, 2:-2, filter_index]
        loss_value = loss(activation)
    
    # if loss_value == 0:
    #     print("Loss is 0")
    
    gradients = tape.gradient(loss_value, image)
    gradients = tf.math.l2_normalize(gradients)
    image += gradients*lr
    return image


# In[8]:


model3.summary()


# In[9]:


model = model3
image_shape = (256, 256)
filter_index = 0
sample_layer = model.get_layer("5th_set_3rd_conv_3x3")
feature_extractor_model = tf.keras.Model(inputs=model.input, outputs=sample_layer.output)
lr=10


# In[10]:


loss_value = 0
while loss_value==0:
    noise_image = tf.random.uniform(minval=0.0, maxval=1.0, shape=(10000,)+image_shape+(3,))
    loss_value = loss(feature_extractor_model(noise_image)[:, :, :, 0])


# In[11]:


index = tf.where(tf.reduce_mean(feature_extractor_model(noise_image)[:, :, :, 0], axis=[1, 2], keepdims=False)).numpy()[0][0]


# In[12]:


loss(feature_extractor_model(tf.expand_dims(noise_image[index], axis=0))[:, :, :, 0])


# In[13]:


del noise_image


# In[14]:


# %%time
# image_shape = (256, 256)

# # models=[model1, model2, model3]
# # models_name = ["model_1_new_version", "model_2_new_version", "model_3_new_version"]

# models=[model3]
# models_name = ["model_3_new_version"]

# epochs=100
# learning_rate = 10

# for model_name, model in zip(models_name, models):
#     for layer in get_convs_layers(model):
#         feature_extractor_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
#         images = []
#         filters_count = layer.output.shape[-1]
#         print(f"layer: {layer.name}")
#         for filter_index in range(filters_count):
#             print(f"filter: index{filter_index}")
#             noise_image = tf.random.uniform(minval=0.4, maxval=0.6, shape=(1,)+image_shape+(3,))
#             for epoch in range(epochs):
#                 noise_image = optimize_step(feature_extractor_model, noise_image, filter_index, learning_rate)
#             images.append(preproces_img(noise_image.numpy()[0].copy()))
#         print_filters(model_name=model_name, layer_name=layer.name, all_images=images)


# In[15]:


130%5==0


# In[16]:


10**-1


# In[17]:


exponent=3
for cicles in range(50):
    exponent *= np.exp((cicles-15)/25)
    exponent = math.ceil(exponent)
    print(exponent)
    print(10**(-exponent))


# In[18]:


get_ipython().run_cell_magic('time', '', 'image_shape = (256, 256)\n\n# models=[model1, model2, model3]\n# models_name = ["model_1_new_version", "model_2_new_version", "model_3_new_version"]\n\nmodels=[model3]\nmodels_name = ["model_3_Alternative_noise_generation_improved"]\n\nepochs=10\nlearning_rate = 10\n\nfor model_name, model in tqdm(zip(models_name, models), total=3):\n    for layer in get_convs_layers(model):\n        feature_extractor_model = tf.keras.Model(inputs=model.input, outputs=layer.output)\n        images = []\n        filters_count = layer.output.shape[-1]\n        \n        print(f"layer: {layer.name}")\n        for filter_index in tqdm(range(filters_count), leave=False):\n            print(f"filter index: {filter_index+1}")\n            exponent = 3\n            cicles = 0\n            # noise_image = tf.random.uniform(minval=0.0, maxval=1.0, shape=(1,)+image_shape+(3,))\n\n            #generate an image that will trigger the filter\'s response\n            print("generating initial image to trigger loss")\n            loss_target_value = 0\n            while loss_target_value < 10**(-exponent):\n                print(f"target value: {10**(-exponent)}")\n                noise_image_options = tf.random.uniform(minval=0.0, maxval=1.0, shape=(10000,)+image_shape+(3,))\n                loss_target_value = loss(feature_extractor_model(noise_image_options)[:, :, :, filter_index])\n                cicles+=1\n\n                print(f"cicles: {cicles}")\n\n                if cicles%5==0:\n                    print(f"Increase exponent by 1 now it\'s: {exponent}")\n                    exponent += 1\n\n                exponent = math.ceil(exponent)\n                print(f"exponent: {exponent}")\n            \n            print(f"loss init value: {loss_target_value}")\n            non_zero_loss_index = tf.where(tf.reduce_mean(feature_extractor_model(noise_image_options)[:, :, :, filter_index], axis=[1, 2], keepdims=False)).numpy()[0][0]\n\n            noise_image = tf.expand_dims(noise_image_options[non_zero_loss_index], axis=0)\n\n            if loss(feature_extractor_model(noise_image)[:, :, :, filter_index]) == 0:\n                print(f"Loss zero for {filter_index}")\n                assert 1!=0\n            \n            print("Optimization step!")\n            for epoch in range(epochs*(exponent)):\n                noise_image = optimize_step(feature_extractor_model, noise_image, filter_index, learning_rate)\n\n            images.append(preproces_img(noise_image.numpy()[0].copy()))\n\n        print_filters(model_name=model_name, layer_name=layer.name, all_images=images)\n')

