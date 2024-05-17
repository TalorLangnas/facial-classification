from matplotlib import pyplot as plt
import tensorflow as tf
from keras import layers
from process import data_rgb, img_height, img_width

# Define model filename
model_filename = f"models/faces_model_logistic_regression_rgb.keras"

"""
Logistic Regression (SoftMax) model
The model has input layer with 222 * 222 * 3 neurons 
output layer with 6 neurons, which is the number of classes.
The output layer uses softmax activation for multi-class classification.

trained on RGB images of size 222x222, pixel values[0, 255].
train = 80% (Validation = 20%), test = 20%
batch_size = 64
epochs = 200
Score:
Training:
  Loss:  687.0675659179688
  Accuracy:  0.7093740701675415
Validation:
  Loss: 2563.52880859375
  Accuracy:  0.3877881169319153
Test:
    Loss: 2618.802001953125
    Accuracy: 0.3671950697898865

    With 200 epochs, we have extreme over Fitting!!


    for 50 epochs:
    Training Loss:  1292.5906982421875
    Training accuracy:  0.5317018628120422

    Validation Loss:  1391.0345458984375
    Validation accuracy:  0.44359079003334045

    Test Loss: 1482.2261962890625
    Test Accuracy: 0.4354577660560608

    for 5 epochs:
    Training Loss:  1490.2591552734375
    Training accuracy:  0.4114672839641571

    Validation Loss:  2565.497314453125
    Validation accuracy:  0.4019409716129303

    Test Loss: 2523.533203125
    Test Accuracy: 0.40537044405937195
    
    With 5 epochs we still have what to learn
    
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_layer (InputLayer)    [(None, 222, 222, 3)]     0

 flatten_layer (Flatten)     (None, 147852)            0

 output_layer (Dense)        (None, 6)                 887118

=================================================================
Total params: 887118 (3.38 MB)
Trainable params: 887118 (3.38 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
"""

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = data_rgb()

    # Define the input layer
    input_shape = (img_height, img_width, 3)  # RGB image with three channels
    input_layer = layers.Input(shape=input_shape, name='input_layer')

    # Flatten the input
    flatten_layer = layers.Flatten(name='flatten_layer')(input_layer)

    # Output layer with softmax activation for multi-class classification
    output_layer = layers.Dense(6, activation='softmax', name='output_layer')(flatten_layer)

    # Create the model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam',  # default learning rate is 0.001
                  loss='categorical_crossentropy',  # For multi-class classification
                  metrics=['accuracy'])

    # Display the model summary
    model.summary()

    # Assuming X_train_gray and y_train are your training data and labels
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Plot training loss and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Accuracy for validation and training data
    print("Validation Loss: ", history.history['val_loss'][-1])
    print("Validation accuracy: ", history.history['val_accuracy'][-1])
    print("Training Loss: ", history.history['loss'][-1])
    print("Training accuracy: ", history.history['accuracy'][-1])

    # Assuming X_test_gray and y_test are your test data and labels
    evaluation_result = model.evaluate(X_test, y_test)

    # Print the evaluation metrics
    print(f"Test Loss: {evaluation_result[0]}")
    print(f"Test Accuracy: {evaluation_result[1]}")

#   model.save(model_filename)

# # Load the model
# loaded_model = keras.models.load_model(model_filename)
# loaded_model.summary()
# evaluation_result = loaded_model.evaluate(X_test, y_test)
# print(f"Loss: {evaluation_result[0]}")
# print(f"Accuracy: {evaluation_result[1]}")
