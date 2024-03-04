from keras.models import Model  # type: ignore
from keras.layers import (  # type: ignore
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    Add,
    MaxPooling2D,
    Reshape,
)


def create_captcha_model(
    input_shape=(192, 64, 3), num_digits=6, num_classes=10
):
    inputs = Input(shape=input_shape)

    x = resnet_base(inputs)
    x = Conv2D(num_classes, (3, 3), padding="same", strides=(1, 2))(x)

    # Assuming the output is (None, 6, 1, 10), we reshape it to (None, 6, 10)
    x = Reshape((num_digits, num_classes))(x)

    # Apply softmax activation to convert logits to probabilities
    x = Activation("softmax")(x)
    model = Model(inputs=inputs, outputs=x)

    # Compile the model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def resnet_base(input_tensor):
    x = Conv2D(64, (7, 7), strides=2, padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

    # First downsampling block
    x = resnet_block(x, 64, strides=2)
    # Additional blocks without downsampling
    x = resnet_block(x, 64, strides=1)

    # Second downsampling block
    x = resnet_block(x, 128, strides=2)
    x = resnet_block(x, 128, strides=1)

    # Third downsampling block
    x = resnet_block(x, 256, strides=2)
    x = resnet_block(x, 256, strides=1)

    return x


def resnet_block(x, filters, kernel_size=(3, 3), strides=2):
    """A ResNet block with two convolutional layers and a shortcut connection.
    """
    shortcut = x

    # First convolution
    x = Conv2D(filters, kernel_size, padding="same", strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Second convolution
    x = Conv2D(filters, kernel_size, padding="same", strides=1)(x)
    x = BatchNormalization()(x)

    # Adding the shortcut connection
    if strides != 1:
        # If strides > 1, we need to downsample the shortcut path to have the
        # same dimensions as the main path
        shortcut = Conv2D(filters, (1, 1), padding="same", strides=strides)(
            shortcut
        )
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)

    return x


def convolutional_base(x):
    x = convolutional_block(x, 32)
    x = convolutional_block(x, 64)
    x = convolutional_block(x, 128)
    x = convolutional_block(x, 256)
    x = convolutional_block(x, 512)
    return x


def convolutional_block(x, filters):
    x = Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(filters, (3, 3), activation="relu", padding="same", strides=2)(
        x
    )
    return x
