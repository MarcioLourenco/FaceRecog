import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import cv2
import numpy as np
import tensorflow as tf
from model import L1Dist
from preprocessing import preprocess_img


def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join("data",'application', 'verification_image')):
        input_img = preprocess_img(os.path.join("data",'application','input_image','input_image.jpg'))
        validation_img = preprocess_img(os.path.join("data",'application', 'verification_image', image))
        
        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(results) > detection_threshold)

    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / len(os.listdir(os.path.join("data",'application', 'verification_image'))) 
    verified = verification > verification_threshold
    return results, verified

def verify_photo():
    model_path = os.path.join("models", 'siamesemodel.h5')
    siamese_model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'L1Dist':L1Dist,
            'BinaryCrossentropy':tf.losses.BinaryCrossentropy
            }
    )

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[120:120+250,200:200+250, :]
        
        cv2.imshow('Verification', frame)
        
        # Verification trigger
        if cv2.waitKey(10) & 0xFF == ord('v'):
            # Save input image to application_data/input_image folder 
    #         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #         h, s, v = cv2.split(hsv)

    #         lim = 255 - 10
    #         v[v > lim] = 255
    #         v[v <= lim] -= 10
            
    #         final_hsv = cv2.merge((h, s, v))
    #         img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

            cv2.imwrite(os.path.join('data','application', 'input_image', 'input_image.jpg'), frame)
            # Run verification
            results, verified = verify(siamese_model, 0.9, 0.5)
            print("----------->", verified, np.squeeze(results))
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    verify_photo()