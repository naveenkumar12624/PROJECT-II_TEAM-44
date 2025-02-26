# PROJECT-II_TEAM-44
# Osteo Synergy – An Automated Bone Fracture Detection System using Deep Learning & Azure ML Studio
      Osteo Synergy aims to enhance medical imaging diagnostics by integrating a deep learning-based system for automated bone fracture detection. Utilizing Azure ML Studio, this project focuses on improving the accuracy and speed of fracture diagnosis, thereby streamlining patient care and reducing the burden on healthcare professionals.
# About
      The Osteo Synergy project integrates advanced deep learning algorithms to develop a bone fracture detection system using medical imaging data. Traditional fracture diagnosis involves manual inspection by radiologists, which can be time-consuming and prone to human error. This project addresses these challenges by implementing a neural network model that automates the identification and classification of bone fractures, ensuring quick and precise diagnostic results.
# Features
  1)Implements advanced Residual Neural Network (ResNet) architecture.
  2)Designed for seamless integration with Azure ML Studio.
  3)High accuracy and reliability.
  4)Rapid processing time for real-time diagnosis.
  5)Comprehensive model training using diverse medical imaging datasets.

# Requirements
  1)Operating System: Requires a 64-bit OS (Windows 10 or Ubuntu) compatible with deep learning frameworks.
  2)Development Environment: Python 3.6 or later for coding and implementation.
  3)Deep Learning Frameworks: TensorFlow for model training.
  4)Image Processing Libraries: OpenCV for efficient image processing.
  5)Version Control: Git for collaborative development.
  6)IDE: VSCode for coding, debugging, and version control.
  7)Additional Dependencies: Includes TensorFlow (version 2.4.1), TensorFlow GPU, OpenCV, and other relevant libraries.
# System Architecture
![image](https://github.com/user-attachments/assets/d07953e2-2f52-4115-a610-505189ec98d4)
# Implementation
```
import os 
import random as rnd import matplotlib.pyplot as plt import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img from tensorflow.keras.optimizers import Adam from tensorflow import keras from tensorflow.keras import layers from tensorflow.keras import utils from tensorflow.keras import models from tensorflow.keras.layers import Conv2D from tensorflow.keras.layers import MaxPool2D from tensorflow.keras.layers import Dense from keras import Sequential 
from tensorflow.keras import regularizers from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input from sklearn.metrics import classification_report,confusion_matrix from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau from tensorflow.keras.applications import VGG16 import tensorflow as tf 
 
# Set GPU memory growth config = tf.compat.v1.ConfigProto() config.gpu_options.allow_growth = True config.log_device_placement = True sess = tf.compat.v1.Session(config=config) tf.compat.v1.keras.backend.set_session(sess) 
 
# Clone the dataset repository 
!git clone https://github.com/naveenkumar12624/BoneData.git 
 
# Adjust the directory paths my_data_dir = '/content/BoneData/Data' train_path = my_data_dir + '/train/' test_path = my_data_dir + '/val/' 
 
# Image parameters 
image_shape = [224, 224] 
 
 
# Data augmentation for training images image_gen = ImageDataGenerator(rotation_range=40, # rotate the image 20 degrees                                width_shift_range=0.10, # Shift the pic width by a max of 5% 

                               height_shift_range=0.10, # Shift the pic height by a max of 5%                                rescale=1/255, # Rescale the image by normalzing it. 
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)                                zoom_range=0.1, # Zoom in by 10% max                                horizontal_flip=True, # Allow horizontal flipping 
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value 
                              ) 
batch_size = 512 
# Flow from directory for training and testing data 
train_image_gen = image_gen.flow_from_directory(     train_path, 
    target_size=image_shape[:2],     color_mode='rgb',     batch_size=batch_size,     seed=32, 
    class_mode='binary'  # Use 'binary' for binary classification 
) 
 
test_image_gen = image_gen.flow_from_directory(     test_path, 
    target_size=image_shape[:2],     color_mode='rgb',     batch_size=batch_size,     class_mode='binary',     shuffle=False 
) 
from tensorflow.keras.applications import ResNet152V2 
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
BatchNormalization, add, Activation 
from tensorflow.keras import Model 
 
# Define ResNet152V2 base model 
base_model = ResNet152V2(input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='avg') 
 
# Freeze the base model layers for layer in base_model.layers: 
    layer.trainable = False 
 
# Define residual block function def residual_block(x, filters, kernel_size=3, stride=1): 
    shortcut = x 
 
    x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)     x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)     x = Activation('relu')(x) 
 
    x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x) 

    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x) 
 
    x = add([x, shortcut])     x = Activation('relu')(x) 
 
    return x 
 
# Create the combined model using Functional API inputs = Input(shape=(224, 224, 3)) x = base_model(inputs) 
 
# Reshape the output from base model x = Flatten()(x) 
x = Dense(256, activation='relu')(x) x = Dropout(0.5)(x) 
 
x = Dense(128, activation='relu')(x) x = Dropout(0.5)(x) 
 
outputs = Dense(1, activation='sigmoid')(x) 
 
# Create the final combined model 
resnet_with_residual = Model(inputs=inputs, outputs=outputs) 
 
resnet_with_residual.summary() 
resnet_with_residual.compile(loss='binary_crossentropy',               optimizer='adam',               metrics=['accuracy']) 
 
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001) 
#learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001) 
early_stopping = EarlyStopping(monitor='val_loss', patience=10) 
 
history = resnet_with_residual.fit(train_image_gen, 
                 validation_data=test_image_gen,                  epochs=20, 
#                class_weight=class_weights,                  callbacks=[learning_rate_reduction],                  verbose=1) 
 
 
 
 

# Save the model 
resnet_with_residual.save("./ResNet50_fracture_model.h5") 
 
pred_probabilities = resnet_with_residual.predict(test_image_gen) predictions = pred_probabilities > 0.5 print(classification_report(test_image_gen.classes,predictions)) print(confusion_matrix(test_image_gen.classes,predictions)) 
  
# Evaluate the model on the test set 
results = resnet_with_residual.evaluate(test_image_gen, verbose=0) print("Test Results:") print("Loss:", results[0]) print("Accuracy:", results[1]) import os import random import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow.keras.preprocessing.image import load_img, img_to_array from tensorflow.keras.models import load_model 
 
# Paths and configurations 
model_path = './ResNet50_fracture_model.h5'  # Update with the path to your saved model data_dir = './BoneData/Data' 
 
# Load the saved model 
model = load_model(model_path) 
 
# Select a random image random_category = random.choice(['fractured', 'not fractured']) random_image_path = os.path.join(data_dir, 'val', random_category, random.choice(os.listdir(os.path.join(data_dir, 'val', random_category)))) 
 
# Load and preprocess the image 
image = load_img(random_image_path, target_size=(224, 224)) image_array = img_to_array(image) image_array = np.expand_dims(image_array, axis=0) image_array /= 255.0  # Normalize the image data 
 
 
# Make predictions 
prediction = model.predict(image_array) 
 
# Display the image plt.imshow(image) 
plt.axis('off') 
plt.title(f"Actual Class: {random_category.capitalize()}\nPredicted Class: {'Fractured' if prediction[0][0] > 0.4 else 'Not Fractured'}") plt.show() 
```

# Output
Detection report with fracture classification.
![image](https://github.com/user-attachments/assets/cdb45584-5c26-469b-8c0c-79c07c38014f)
![image](https://github.com/user-attachments/assets/1949c24e-8db6-4b9b-9b53-dd1b4a06ddbd)

Visual representation of detected fractures on medical images.
![image](https://github.com/user-attachments/assets/24bcf630-c313-49ff-afac-b5177f74d2e3)
![image](https://github.com/user-attachments/assets/5368ea9c-5c08-4d46-9024-4c77aaedb1f0)

Detection Accuracy: 94% with a validation accuracy of 80%.
# Results and Impact
    The Osteo Synergy system enhances the efficiency of bone fracture diagnosis, providing rapid and accurate results. This technology reduces the workload on radiologists and healthcare providers while ensuring patients receive timely and precise diagnoses. The project's success demonstrates the potential for integrating deep learning in medical diagnostics, paving the way for future advancements in healthcare technology.

# Articles published / References
    [1]	SHARMA, GAURAV, ET AL. "BONE FRACTURE DETECTION USING DEEP LEARNING: A REVIEW." IN 2020 IEEE 17TH INTERNATIONAL SYMPOSIUM ON BIOMEDICAL IMAGING (ISBI), PP. 1671-1675. IEEE, 2020. 
    [2]	GULSHAN, VARUN, ET AL. "AUTOMATED BONE FRACTURE DETECTION AND LOCALIZATION USING DEEP LEARNING." IN 2018 IEEE EMBS INTERNATIONAL CONFERENCE ON BIOMEDICAL & HEALTH INFORMATICS (BHI), PP. 269-272. IEEE, 2018. 
    [3]	HOU, LI, ET AL. "BONE FRACTURE DETECTION VIA DEEP LEARNING." IN 2019 IEEE INTERNATIONAL CONFERENCE ON BIOINFORMATICS AND BIOMEDICINE (BIBM), PP. 2516-2521. IEEE, 2019. 
    [4]	KHAN, A. I., ET AL. "AUTOMATED BONE FRACTURE DETECTION SYSTEM USING CONVOLUTIONAL NEURAL NETWORKS." IN 2020 IEEE 22ND 
INTERNATIONAL CONFERENCE ON E-HEALTH NETWORKING, APPLICATIONS AND SERVICES (HEALTHCOM), PP. 1-6. IEEE, 2020. 
    [5]	REN, SHUAI, ET AL. "AN EFFICIENT BONE FRACTURE DETECTION APPROACH USING DEEP LEARNING." IN 2019 IEEE INTERNATIONAL CONFERENCE ON MULTIMEDIA & EXPO WORKSHOPS (ICMEW), PP. 602-607. IEEE, 2019. 
    [6]	BAJAJ, HIMANI, ET AL. "BONE FRACTURE DETECTION USING TRANSFER LEARNING AND DEEP NEURAL NETWORKS." IN 2021 IEEE REGION 10 SYMPOSIUM (TENSYMP), PP. 1910-1915. IEEE, 2021. 
    [7]	NOUR, MOHAMED, ET AL. "DEEP LEARNING-BASED SYSTEM FOR BONE FRACTURE DETECTION AND LOCALIZATION." IN 2019 IEEE EMBS 
INTERNATIONAL CONFERENCE ON BIOMEDICAL & HEALTH INFORMATICS (BHI), PP. 1-4. IEEE, 2019.  
    [8]	ZHOU, XIAOBO, ET AL. "DEEP LEARNING FOR BONE FRACTURE DETECTION ON CHEST RADIOGRAPHS." IN 2020 42ND ANNUAL INTERNATIONAL CONFERENCE OF THE IEEE ENGINEERING IN MEDICINE & BIOLOGY SOCIETY (EMBC), PP. 40374040. IEEE, 2020. 
    [9]	PAN, SHILPA, ET AL. "BONE FRACTURE DETECTION IN X-RAY IMAGES USING DEEP LEARNING TECHNIQUES." IN 2021 IEEE 6TH INTERNATIONAL CONFERENCE FOR CONVERGENCE IN TECHNOLOGY (I2CT), PP. 1-6. IEEE, 2021. 
    [10]	SINGH, S., ET AL. "A COMPREHENSIVE REVIEW ON BONE FRACTURE DETECTION TECHNIQUES USING DEEP LEARNING." IN 2023 IEEE INTERNATIONAL 
CONFERENCE ON BIOINFORMATICS AND BIOMEDICINE (BIBM), PP. 1-5. IEEE, 2023
