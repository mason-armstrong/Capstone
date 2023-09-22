import cv2
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.applications.vgg16 import VGG16
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.random.set_seed(42)

#Load Image
image_path = "Training Data\\PigweedDataSet\\annotated_images\\pigweed_001.png"
image = cv2.imread(image_path)
if image is None:
    print("Could not read image")
    exit()

#Reize and preprocess image
image = cv2.resize(image, (224, 224))
image = image.astype('float32')
image /= 255.0


#Load pre-trained Vgg16 model and higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

#Add custom layers
x = base_model.output
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dense(4096, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x) #Assumes 5 classes

#create final model
model = keras.Model(inputs=base_model.input, outputs=predictions)

#Fine tune the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Predict the class of the image
image_batch = np.expand_dims(image, axis=0)
result = model.predict(image_batch)
print("Class prediction vector [P(class1), P(class2), ...]: ", result)
print("Predicted class: ", np.argmax(result))





