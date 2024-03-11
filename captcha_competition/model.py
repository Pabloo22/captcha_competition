def create_captcha_model(
    input_shape=(192, 64, 3), num_digits=6, num_classes=10, conv_base="resnet"
):
    from keras.models import Model  # type: ignore
    from keras.layers import (  # type: ignore
        Input,
        Conv2D,
        Activation,
        Reshape,
    )

    inputs = Input(shape=input_shape)

    x = convolutional_base_factory(conv_base)(inputs)
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


def convolutional_base_factory(conv_base: str):
    conv_bases = {
        "resnet": resnet_base,
        "efficientnet": efficientnet_base,
    }
    conv_base = conv_base.lower()
    if conv_base in conv_bases:
        return conv_bases[conv_base]

    raise ValueError(
        f"Unknown conv_base: {conv_base}. Available: {conv_bases.keys()}"
    )


def resnet_base(input_tensor):
    from keras.layers import (  # type: ignore
        Conv2D,
        BatchNormalization,
        Activation,
        MaxPooling2D,
    )

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


def efficientnet_base(input_tensor):
    from keras.layers import (  # type: ignore
        Conv2D,
        BatchNormalization,
        ReLU,
    )

    # Initial Convolution
    x = Conv2D(32, kernel_size=3, strides=2, padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Downsampling through MBConv blocks
    x = mb_conv_block(x, filter_num=16, kernel_size=3, strides=2)
    x = mb_conv_block(x, filter_num=24, kernel_size=3, strides=2)
    x = mb_conv_block(x, filter_num=32, kernel_size=3, strides=2)
    x = mb_conv_block(x, filter_num=64, kernel_size=3, strides=2)
    x = mb_conv_block(x, filter_num=96, kernel_size=3, strides=2)

    return x


def resnet_block(x, filters, kernel_size=(3, 3), strides=2):
    """A ResNet block with two convolutional layers and a shortcut connection."""
    from keras.layers import (  # type: ignore
        Conv2D,
        BatchNormalization,
        Activation,
        Add,
    )

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


def mb_conv_block(inputs, filter_num, kernel_size, strides):
    """Mobile Inverted Bottleneck Convolution block."""
    from keras.layers import (  # type: ignore
        Conv2D,
        BatchNormalization,
        ReLU,
        DepthwiseConv2D,
    )

    x = Conv2D(filter_num, kernel_size=1, strides=1, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = DepthwiseConv2D(
        kernel_size=kernel_size, strides=strides, padding="same"
    )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filter_num, kernel_size=1, strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    return x
