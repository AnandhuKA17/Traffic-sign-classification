import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score

data = []
labels = []
classes = 43
cur_path = os.getcwd()


# Retrieving the images and their labels
for i in range(classes):
    path = os.path.join(cur_path, r'C:\Users\ACER\Desktop\Traffic sign classification\archive_project\Train', str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image_path = os.path.join(path, a)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

            if image is None:
                print(f"Error loading image: {image_path}")
                continue

            image = cv2.resize(image, (30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(f"Error processing image: {image_path}, Error: {e}")

# Converting lists into numpy arrays
data = np.array(data) / 255.0
labels = np.array(labels)

print(data.shape, labels.shape)
# Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Converting the labels into one hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# Build the original Sequential model
model = Sequential()

# Convolutional layers
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(30, 30, 3)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten layer
model.add(Flatten())

# Fully connected layers
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(43, activation='softmax'))

# Compilation of the model
model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Print the model summary
model.summary()

# Training the original model
epochs = 25
batch_size = 32

# Define callbacks for early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('model.h5', save_best_only=True)

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, model_checkpoint]
)

# Plotting graphs for accuracy
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('loss_plot.png')
# Testing accuracy on the test dataset
test_data = pd.read_csv(r'C:\Users\ACER\Desktop\Traffic sign classification\archive_project\Test.csv')

labels = test_data["ClassId"].values
imgs = test_data["Path"].values

data = []

for img in imgs:
    path1 = 'C:/Users/ACER/Desktop/Traffic sign classification/archive_project/'
    path_new = os.path.join(path1, img)
    try:
        image = cv2.imread(path_new)
        if image is None:
            print(f"Error loading image: {img}")
            continue
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (30, 30))
        image = np.array(image)
        data.append(image)
    except Exception as e:
        print(f"Error processing image: {img}, Error: {e}")

X_test = np.array(data) / 255.0

pred = model.predict(X_test)
test_loss, test_accuracy = model.evaluate(X_test, to_categorical(labels, 43))
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Convert predictions to class labels
predicted_labels = np.argmax(pred, axis=1)
# Accuracy with the test data
print(accuracy_score(labels, predicted_labels))

# Class definitions
classes = {1: 'Speed limit (20km/h)', 2: 'Speed limit (30km/h)', 3: 'Speed limit (50km/h)', 4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)', 6: 'Speed limit (80km/h)', 7: 'End of speed limit (80km/h)',
           8: 'Speed limit (100km/h)', 9: 'Speed limit (120km/h)', 10: 'No passing',
           11: 'No passing veh over 3.5 tons', 12: 'Right-of-way at intersection', 13: 'Priority road', 14: 'Yield',
           15: 'Stop', 16: 'No vehicles', 17: 'Veh > 3.5 tons prohibited', 18: 'No entry', 19: 'General caution',
           20: 'Dangerous curve left', 21: 'Dangerous curve right', 22: 'Double curve', 23: 'Bumpy road',
           24: 'Slippery road', 25: 'Road narrows on the right', 26: 'Road work', 27: 'Traffic signals',
           28: 'Pedestrians', 29: 'Children crossing', 30: 'Bicycles crossing', 31: 'Beware of ice/snow',
           32: 'Wild animals crossing', 33: 'End speed + passing limits', 34: 'Turn right ahead', 35: 'Turn left ahead',
           36: 'Ahead only', 37: 'Go straight or right', 38: 'Go straight or left', 39: 'Keep right', 40: 'Keep left',
           41: 'Roundabout mandatory', 42: 'End of no passing', 43: 'End no passing veh > 3.5 tons' }

# Example Prediction
def prediction_new(file_name):
    image = cv2.imread(file_name)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (30, 30))
    image_array = np.array(image)
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)
    sign = classes[predicted_class]
    print(f"Predicted Class: {sign}")

prediction_new(r'C:\Users\ACER\Desktop\Traffic sign classification\archive_project\Test\00006.png')