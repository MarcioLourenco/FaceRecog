import sys
sys.stdout.reconfigure(encoding='utf-8')

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

from model import make_siamese_model, make_embedding
from preprocessing import preprocess_data

tf.config.run_functions_eagerly(True)


def reload_model(LEARNING_RATE=1e-4, embedding_dim=100):

    checkpoint_dir = 'data/training_checkpoints'
    embedding = make_embedding(embedding_dim)
    siamese_model = make_siamese_model(embedding_dim, embedding)
    opt = tf.keras.optimizers.Adam(LEARNING_RATE) 
    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    siamese_model.save('models/siamesemodel.h5')

    return siamese_model

def evaluate_model(batch_size=128):
    siamese_model = reload_model()
    train_data, test_data = preprocess_data(batch_size)

    recall = Recall()
    precision = Precision()
    acc = BinaryAccuracy()

    for test_input, test_val, y_true in test_data.as_numpy_iterator():
        yhat = siamese_model.predict([test_input, test_val])
        yhat_binary = (yhat >= 0.5).astype(int)

        recall.update_state(y_true, yhat_binary)
        precision.update_state(y_true,yhat_binary) 
        acc.update_state(y_true, yhat_binary)

    print(f"Precision: {precision.result().numpy():.2f}")
    print(f"Recall: {recall.result().numpy():.2f}")
    print(f"Accuracy: {acc.result().numpy():.2f}")

evaluate_model()