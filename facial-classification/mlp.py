from process import get_data
import keras
from keras import layers

# Define model filename
model_filename = f"models/faces_model_mlp.keras"

"""
Create a simple multi-layer perceptron (MLP) model
The model has 4 hidden layers with 500, 250, 64, and 6 neurons, respectively.

The model trained on grayscale images of size 222x222.
train = 80%, test = 20%
batch_size = 128
epochs = 15
score:
    Loss: 1.18295419216156
    Accuracy: 0.5257198214530945


Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 flatten (Flatten)           (None, 49284)             0

 dense (Dense)               (None, 500)               24642500

 dense_1 (Dense)             (None, 250)               125250

 dense_2 (Dense)             (None, 64)                16064

 dense_3 (Dense)             (None, 6)                 390

=================================================================
Total params: 24784204 (94.54 MB)
Trainable params: 24784204 (94.54 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
"""

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()

    input_shape = (222, 222, 1)
    hidden_layer_1_size = 500
    hidden_layer_2_size = 250
    hidden_layer_3_size = 64
    num_classes = 6

    # Build the model:
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(hidden_layer_1_size, activation="relu"),
            layers.Dense(hidden_layer_2_size, activation="relu"),
            layers.Dense(hidden_layer_3_size, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    batch_size = 128
    epochs = 10

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # Save the model
    model.save(model_filename)

    # # Load the model
    # loaded_model = keras.models.load_model(model_filename)  
    # loaded_model.summary()
    # evaluation_result = loaded_model.evaluate(X_test, y_test)
    # print(f"Loss: {evaluation_result[0]}")
    # print(f"Accuracy: {evaluation_result[1]}")
