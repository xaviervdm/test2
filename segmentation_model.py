import tensorflow as tf
from tensorflow.keras import layers, models

def conv_block(inputs, num_filters):
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape=(256, 256, 3), num_classes=1):
    """Builds a simple U-Net model for semantic segmentation.

    Args:
        input_shape: Input image shape. Defaults to (256, 256, 3).
        num_classes: Number of output classes. Defaults to 1.

    Returns:
        A tf.keras Model instance representing the U-Net.
    """
    inputs = layers.Input(shape=input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    activation = 'softmax' if num_classes > 1 else 'sigmoid'
    outputs = layers.Conv2D(num_classes, 1, padding='same', activation=activation)(d4)

    model = models.Model(inputs, outputs, name="unet")
    return model


def train_unet(train_dataset,
              val_dataset=None,
              epochs=10,
              num_classes=1,
              model_path=None):
    """Compiles and trains a U-Net model.

    Args:
        train_dataset: `tf.data.Dataset` yielding `(image, mask)` pairs for training.
        val_dataset: Optional dataset for validation.
        epochs: Number of training epochs. Defaults to 10.
        num_classes: Number of output classes. Defaults to 1.
        model_path: Optional path to save the trained model.

    Returns:
        The trained `tf.keras` model.
    """
    model = build_unet(num_classes=num_classes)
    loss = "binary_crossentropy" if num_classes == 1 else "sparse_categorical_crossentropy"
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
    if model_path:
        model.save(model_path)
    return model

if __name__ == "__main__":
    model = build_unet()
    model.summary()
