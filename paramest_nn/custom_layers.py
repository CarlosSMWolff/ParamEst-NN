import tensorflow as tf
from tensorflow import keras
import numpy as np

def theta(x):
  return tf.experimental.numpy.heaviside(x,1)

def sigmoid(x, factor = 20.):
  return tf.nn.sigmoid(factor*x)

def theta_prod(x, width = 1.):
  return theta(x+width/2)*theta(-x+width/2)

def theta_prod_sigmoid(x, factor = 20., width = 1.):
  return sigmoid(x+width/2,factor)*sigmoid(-x+width/2,factor)

# Clear all previously registered custom objects
keras.saving.get_custom_objects().clear()

#@tf.function
@keras.saving.register_keras_serializable(name="histogram_sigmoid")
def histogram_sigmoid(input_tensor, nbins,shifts, width):
  'I made this to have smoother, differentiable functions '
  input_tensor = tf.expand_dims(input_tensor, axis=0)
  units = nbins
  onesies = tf.ones([units,1])

  input_copy = tf.tensordot(onesies, input_tensor, axes = [[1], [0]])
  shifts = tf.tensordot(shifts, tf.math.divide(input_tensor,input_tensor), axes = [[1], [0]])

  input_shifted = input_copy - shifts

  factor = 20
  input_theta= tf.nn.sigmoid((input_shifted+width/2)*factor)*tf.nn.sigmoid((-input_shifted+width/2)*factor)

  #input_theta=theta_prod_sigmoid(input_shifted, width = width)
  histogram = tf.transpose(tf.math.reduce_sum(input_theta,axis=-1))
  return histogram


  ### SIGMOID HISTOGRAM LAYER ###
# Using sigmoids I have well-defined gradients.

@keras.saving.register_keras_serializable()
class MyHistogramLayer_Sigmoid(keras.layers.Layer):
    def __init__(self, nbins=399,taumax=100., trainable = True, negative = 0):
        super().__init__()
        self.taumax = taumax
        self.nbins = nbins
        self.width_0 = taumax/nbins
        self.trainable = trainable
        self.negative = negative

    def build(self, batch_input_shape):
        self.width = self.add_weight(name = 'width',
            shape=([1]),
            initializer=tf.constant_initializer([self.width_0]),
            trainable=self.trainable,
        )
        self.shifts = self.add_weight(name = 'shifts',
            shape=([self.nbins, 1]),
            initializer=tf.constant_initializer(
                np.reshape(np.linspace(-self.negative*(self.taumax)+self.width_0/2,self.taumax-self.width_0/2,self.nbins),[self.nbins,1])
                ),
            trainable=self.trainable        )


    def call(self, inputs):
        return histogram_sigmoid(inputs, self.nbins, self.shifts, self.width)

    def get_config(self):
      return {"nbins": self.nbins, "taumax": self.taumax, "trainable":self.trainable, "negative" : self.negative}
    
    
    # Custom Layer
@keras.saving.register_keras_serializable()
class ExpandDimLayer(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ExpandDimLayer, self).__init__(**kwargs)
        self.axis = axis
    
    def build(self, input_shape):
        super(ExpandDimLayer, self).build(input_shape)
    
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)
    
    def get_config(self):
        config = super(ExpandDimLayer, self).get_config()
        config.update({'axis': self.axis})
        return config