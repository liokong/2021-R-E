import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

X_test = np.load('input1.npy')

model = tf.keras.models.load_model('unet_mse.h5')

preds=model.predict(X_test)
preds=(preds>0.5).astype(np.uint8)

np.save('spine.npy', preds)

print(preds)