import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.metrics import Precision, Recall


from model import make_siamese_model, make_embedding
from preprocessing import preprocess_data

def train_model(EPOCHS=50, LEARNING_RATE=1e-4, embedding_dim=100, batch_size=128):
    def create_checkpoint_folder():
        checkpoint_dir = 'data\\training_checkpoints' 
        checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')   
        if not os.path.exists(checkpoint_prefix):
                os.makedirs(checkpoint_prefix)
        return checkpoint_prefix

    train_data, test_data = preprocess_data(batch_size)
    binary_cross_loss = tf.losses.BinaryCrossentropy()
    opt = tf.keras.optimizers.Adam(LEARNING_RATE) # 0.0001
    embedding = make_embedding(embedding_dim)
    siamese_model = make_siamese_model(embedding_dim, embedding)
    checkpoint_prefix = create_checkpoint_folder()
    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)


    @tf.function
    def train_step(batch):
        
        # Record all of our operations 
        with tf.GradientTape() as tape:     
            # Get anchor and positive/negative image
            X = batch[:2]
            # Get label
            y = batch[2]
            
            # Forward pass
            yhat = siamese_model(X, training=True)
            # Calculate loss
            loss = binary_cross_loss(y, yhat)
        print(loss)
            
        # Calculate gradients
        grad = tape.gradient(loss, siamese_model.trainable_variables)
        
        # Calculate updated weights and apply to siamese model
        opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
            
        # Return loss
        return loss

    def train_chunck(data, EPOCHS):
        
        # Loop through epochs
        for epoch in range(1, EPOCHS+1):
            print('\n Epoch {}/{}'.format(epoch, EPOCHS))
            progbar = tf.keras.utils.Progbar(len(data))
            
            # Creating a metric object 
            r = Recall()
            p = Precision()
            
            # Loop through each batch
            for idx, batch in enumerate(data):
                # Run train step here
                loss = train_step(batch)
                yhat = siamese_model.predict(batch[:2])
                r.update_state(batch[2], yhat)
                p.update_state(batch[2], yhat) 
                progbar.update(idx+1)
            print(loss.numpy(), r.result().numpy(), p.result().numpy())
            
            # Save checkpoints
            if epoch % 10 == 0: 
                checkpoint.save(file_prefix=checkpoint_prefix)

    train_chunck(train_data, EPOCHS)
    siamese_model.save("data/siamesemodel.h5")

if __name__ == "__main__":
    train_model(EPOCHS=25)