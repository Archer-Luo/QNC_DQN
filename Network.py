import tensorflow as tf
import ray
from tensorflow.python import keras


@ray.remote
class Network(object):
    def __init__(self, n_actions):
        dqn = keras.Sequential([
            keras.layers.Dense(10, input_dim=2, activation='tanh'),
            keras.layers.Dense(10, activation='tanh'),
            keras.layers.Dense(10, activation='tanh'),
            keras.layers.Dense(n_actions)
        ])
        dqn.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam')
        self.model = dqn

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def predict(self, states):
        return self.model.predict(states)

    def call(self, states):
        return self.model(states)

    def trainable_variables(self):
        return self.model.trainable_variables

    def apply_gradients(self, model_gradients, trainable_variables):
        self.model.optimizer.apply_gradients(zip(model_gradients, trainable_variables))

    def get_model(self):
        return self.model
