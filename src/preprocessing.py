
import os
import tarfile
import tensorflow as tf
import cv2
import uuid
import numpy as np
from matplotlib import pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')


def create_folders():
    if not os.path.exists(POS_PATH):
        os.makedirs(POS_PATH)

    if not os.path.exists(NEG_PATH):
        os.makedirs(NEG_PATH)

    if not os.path.exists(ANC_PATH):
        os.makedirs(ANC_PATH)


def get_negative_data():
    file_path = "data/lfw.tgz"
    output_dir = "data"   

    if os.path.isfile(file_path):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=output_dir)

    for directory in os.listdir("data/lfw"):
        for file in os.listdir(os.path.join("data/lfw", directory)):
            EX_PATH = os.path.join("data/lfw", directory, file)
            NEW_PATH = os.path.join(NEG_PATH, file)
            os.replace(EX_PATH, NEW_PATH)


def get_anchor_positive_data():
    os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))

    cap = cv2.VideoCapture(0)
    while cap.isOpened(): 
        ret, frame = cap.read()
    
        frame = frame[120:120+250,200:200+250, :]
        
        if cv2.waitKey(1) & 0XFF == ord('a'):
            imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, frame)
        
        if cv2.waitKey(1) & 0XFF == ord('p'):
            imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, frame)
        
        cv2.imshow('Image Collection', frame)
        
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


def data_augmentation(img, n):
    data = []
    for i in range(n -1):
        img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1,2))
        img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1,3))
        # img = tf.image.stateless_random_crop(img, size=(20,20,3), seed=(1,2))
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9,upper=1, seed=(np.random.randint(100),np.random.randint(100)))
            
        data.append(img)
    
    return data


def image_augmentation(PATH):
    for file_name in os.listdir(os.path.join(PATH)):
        img_path = os.path.join(PATH, file_name)
        img = cv2.imread(img_path)
        # plt.imshow(img)
        augmented_images = data_augmentation(img, 10) 
        
        for image in augmented_images:
            cv2.imwrite(os.path.join(PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())


def preprocess_img(file_path):
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    # Scale image to be between 0 and 1 
    img = img / 255.0

    # Return image
    return img


def preprocess_twin(input_img, validation_img, label):
    return(preprocess_img(input_img), preprocess_img(validation_img), label)


def preprocess_data():
    anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(3000)
    positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(3000)
    negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(3000)

    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    data = positives.concatenate(negatives)

    data = data.map(preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=10000)

    train_data = data.take(round(len(data)*.7))
    train_data = train_data.batch(16)
    train_data = train_data.prefetch(8)

    test_data = data.skip(round(len(data)*.7))
    test_data = test_data.take(round(len(data)*.3))
    test_data = test_data.batch(16)
    test_data = test_data.prefetch(8)
    return train_data, test_data


if __name__ == "__main__":
    # create_folders()
    # get_negative_data()
    # get_anchor_positive_data()
    # image_augmentation(POS_PATH)
    # image_augmentation(ANC_PATH)
    train_data, test_data = preprocess_data()

