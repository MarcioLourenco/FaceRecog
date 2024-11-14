import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten

tf.config.run_functions_eagerly(True)

def make_embedding(input_size): 
    inp = Input(shape=(input_size,input_size,3), name='input_img')
    
    # First block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    # Third block 
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')


# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        input_embedding = input_embedding[0] if isinstance(input_embedding, list) else input_embedding
        validation_embedding = validation_embedding[0] if isinstance(validation_embedding, list) else validation_embedding
        return tf.math.abs(input_embedding - validation_embedding)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    

def make_siamese_model(input_size, embedding): 
    
    # Anchor image input in the network
    input_img = Input(name='input_img', shape=(input_size,input_size,3))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(input_size,input_size,3))
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    inp_embedding = embedding(input_img)
    val_embedding = embedding(validation_image)
    distances = siamese_layer(inp_embedding, val_embedding)
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_img, validation_image], outputs=classifier, name='SiameseNetwork')

if __name__ == "__main__":
    embedding = make_embedding(100)
    siamese_model = make_siamese_model(100, embedding)
    print(siamese_model.summary())
