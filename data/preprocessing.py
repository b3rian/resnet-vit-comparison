import tensorflow as tf

def get_model_preprocessing_layer():
    """Returns a preprocessing layer for the model."""
    return tf.keras.Sequential([
        tf.keras.layers.Resizing(64, 64),  # Keep target size (Tiny ImageNet size)
        tf.keras.layers.Rescaling(1./255),  # Scale [0, 255] → [0, 1]
        tf.keras.layers.Normalization(
            mean=[0.485, 0.456, 0.406],
            variance=[0.229**2, 0.224**2, 0.225**2],
            name="imagenet_norm"
        )
    ], name="preprocessing_pipeline")
