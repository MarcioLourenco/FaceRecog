import os
import numpy as np
import tensorflow as tf
from model import L1Dist
from preprocessing import preprocess_img


def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join("data",'application', 'verification_image')):
        input_img = preprocess_img(os.path.join("data",'application','input_image','input_image.jpg'))
        validation_img = os.path.join("data",'application', 'verification_image', image)
        
        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(results) > detection_threshold)

    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / len(os.listdir(os.path.join("data",'application', 'verification_image'))) 
    verified = verification > verification_threshold
    return results, verified

model_path = os.path.join("models", 'siamesemodel.h5')
siamese_model = tf.keras.models.load_model(model_path, 
                                   custom_objects={'L1Dist':L1Dist,
                                                    'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

