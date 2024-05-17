import os
from process import data_rgb, img_height, img_width
import keras
from keras import layers
import tensorflow as tf

# Define model filename
model_filename = f"models/faces_model_mlp_rgb.keras"

"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 flatten (Flatten)           (None, 147852)            0

 dense (Dense)               (None, 500)               73926500

 dense_1 (Dense)             (None, 250)               125250

 dense_2 (Dense)             (None, 64)                16064

 dense_3 (Dense)             (None, 6)                 390

=================================================================
Total params: 74068204 (282.55 MB)
Trainable params: 74068204 (282.55 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
Training the Model on RGB images of size 222x222, pixel values[0, 255].
train = 80%, test = 20%
batch_size = 128
epochs = 15

Validation Loss:  20.122346878051758
Validation accuracy:  0.36863377690315247
Training Loss:  15.48425579071045
Training accuracy:  0.43244943022727966
Test loss: 20.883813858032227
Test accuracy: 0.33872532844543457

"""

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = data_rgb()
    # Check if the model file exists
    # if (False):
    if os.path.exists(model_filename):
        # Load the existing model
        model = tf.keras.models.load_model(model_filename)
        print("Loaded existing model")
    else:

        input_shape = (img_height, img_width, 3)
        # hidden_layer_1_size = 256
        # hidden_layer_2_size = 128
        # hidden_layer_3_size = 32
        # num_classes = 6
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
        epochs = 15

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    # # Plot training loss and validation loss
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    # Accuracy for validation and training data
    print("Validation Loss: ", history.history['val_loss'][-1])
    print("Validation accuracy: ", history.history['val_accuracy'][-1])
    print("Training Loss: ", history.history['loss'][-1])
    print("Training accuracy: ", history.history['accuracy'][-1])

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # # Save the model
    # model.save(model_filename)

    # # Load the model
    # loaded_model = keras.models.load_model(model_filename)
    # loaded_model.summary()
    # evaluation_result = loaded_model.evaluate(X_test, y_test)
    # print(f"Loss: {evaluation_result[0]}")
    # print(f"Accuracy: {evaluation_result[1]}")
