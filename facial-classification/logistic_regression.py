import tensorflow as tf
from keras import layers
from process import get_data

# Define model filename
model_filename = f"models/faces_model_logistic_regression.keras"

"""
Logistic Regression (SoftMax) model
The model has input layer with 222 * 222 neurons 
output layer with 6 neurons, which is the number of classes.
The output layer uses softmax activation for multi-class classification.

trained on grayscale images of size 222x222.
train = 80% (Validation = 20%), test = 20%
batch_size = 64
epochs = 200
Score:
Training:
  Loss:  0.563815176486969
  Accuracy:  0.8009373545646667
Validation:
  Loss: 2.9170939922332764
  Accuracy:  0.38856014609336853
Test:
    Loss: 2.9384377002716064
    Accuracy: 0.3665480315685272

_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_layer (InputLayer)    [(None, 222, 222, 1)]     0

 flatten_layer (Flatten)     (None, 49284)             0

 output_layer (Dense)        (None, 6)                 295710

=================================================================
Total params: 295710 (1.13 MB)
Trainable params: 295710 (1.13 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
"""

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()

    # Define the input layer
    input_shape = (222, 222, 1)  # Grayscale image with one channel
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

    model.save(model_filename)

    # # Load the model
    # loaded_model = keras.models.load_model(model_filename)
    # loaded_model.summary()
    # evaluation_result = loaded_model.evaluate(X_test, y_test)
    # print(f"Loss: {evaluation_result[0]}")
    # print(f"Accuracy: {evaluation_result[1]}")
