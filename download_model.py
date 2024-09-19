import tensorflow_hub as hub
import tensorflow as tf
import os

model_dir = 'saved_model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_url = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
model = hub.load(model_url)

tf.saved_model.save(model, model_dir)

print(f'Model saved to {model_dir}')
