# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns

# Data paths
good_images_path = '/path/to/good/folder'  # replace with the actual path
defective_images_path = '/path/to/defective/folder'  # replace with the actual path

# Image parameters
img_height, img_width = 150, 150
batch_size = 32

# Image Data Generators for preprocessing and augmentation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Train and validation generators
train_generator = train_datagen.flow_from_directory(
    directory=os.path.dirname(good_images_path),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training')  # Set for training data

validation_generator = train_datagen.flow_from_directory(
    directory=os.path.dirname(good_images_path),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation')  # Set for validation data

# Model architecture: Convolutional Neural Network (CNN)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# Evaluate the model on validation data
val_loss, val_acc = model.evaluate(validation_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_acc}")

# Visualize training results
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()

# Confusion matrix and classification report
validation_generator.reset()
Y_pred = model.predict(validation_generator)
y_pred = (Y_pred > 0.5).astype(int)

print('Confusion Matrix')
cm = confusion_matrix(validation_generator.classes, y_pred)
sns.heatmap(cm, annot=True, fmt='d')

print('Classification Report')
print(classification_report(validation_generator.classes, y_pred, target_names=['Good', 'Defective']))

# Save the model for future use
model.save('faultfindy_model.h5')

# Import necessary libraries
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam

# Define a model-building function for hyperparameter tuning
def build_model(hp):
    model = Sequential()

    # First Conv layer with variable number of filters and kernel size
    model.add(Conv2D(filters=hp.Choice('conv_1_filter', values=[32, 64, 128]),
                     kernel_size=hp.Choice('conv_1_kernel', values=[(3, 3), (5, 5)]),
                     activation='relu',
                     input_shape=(img_height, img_width, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second Conv layer
    model.add(Conv2D(filters=hp.Choice('conv_2_filter', values=[64, 128]),
                     kernel_size=hp.Choice('conv_2_kernel', values=[(3, 3), (5, 5)]),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third Conv layer
    model.add(Conv2D(filters=hp.Choice('conv_3_filter', values=[128, 256]),
                     kernel_size=hp.Choice('conv_3_kernel', values=[(3, 3), (5, 5)]),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Dense layer with variable number of units
    model.add(Dense(units=hp.Int('dense_units', min_value=128, max_value=512, step=32), activation='relu'))

    # Dropout layer with variable dropout rate
    model.add(Dropout(rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(Dense(1, activation='sigmoid'))

    # Compile the model with variable learning rate
    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

# Instantiate the tuner object for RandomSearch
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # Number of hyperparameter configurations to try
    executions_per_trial=1,  # Number of models to build and evaluate per configuration
    directory='faultfindy_tuning',  # Directory to save results
    project_name='faultfindy_cnn'
)

# Print a summary of the search space
tuner.search_space_summary()

# Perform the hyperparameter search
tuner.search(train_generator, epochs=10, validation_data=validation_generator)

# Retrieve the best hyperparameters and model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]

# Display the best hyperparameters
print(f"Best number of conv_1 filters: {best_hps.get('conv_1_filter')}")
print(f"Best kernel size for conv_1: {best_hps.get('conv_1_kernel')}")
print(f"Best dropout rate: {best_hps.get('dropout_rate')}")
print(f"Best dense units: {best_hps.get('dense_units')}")
print(f"Best learning rate: {best_hps.get('learning_rate')}")

# Train the best model on full data
best_model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Evaluate the best model
val_loss, val_acc = best_model.evaluate(validation_generator)
print(f"Best Model Validation Loss: {val_loss}")
print(f"Best Model Validation Accuracy: {val_acc}")

