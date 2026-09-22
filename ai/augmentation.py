import tensorflow as tf
from tensorflow.keras import layers

class GaussianNoiseLayer(layers.Layer):
    """Custom Keras Layer to add Gaussian Noise during training."""
    def __init__(self, stddev=0.05, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(
                shape=tf.shape(inputs),
                mean=0.0,
                stddev=self.stddev,
                dtype=inputs.dtype
            )
            # Keep values in range [0, 1]
            return tf.clip_by_value(inputs + noise, 0.0, 1.0)
        return inputs

class ColorAugmentationLayer(layers.Layer):
    """Custom Keras Layer to apply Random Saturation and Hue during training."""
    def __init__(self, max_delta=0.08, lower_sat=0.8, upper_sat=1.2, **kwargs):
        super().__init__(**kwargs)
        self.max_delta = max_delta
        self.lower_sat = lower_sat
        self.upper_sat = upper_sat

    def call(self, inputs, training=None):
        if training:
            # tf.image operations expect inputs without batch dimension or can work with batches
            # Let's map over batch if inputs has rank 4, or process directly
            def augment_single_image(img):
                img = tf.image.random_saturation(img, lower=self.lower_sat, upper=self.upper_sat)
                img = tf.image.random_hue(img, max_delta=self.max_delta)
                return tf.clip_by_value(img, 0.0, 1.0)
            
            # Map across batch size
            augmented = tf.map_fn(augment_single_image, inputs)
            return augmented
        return inputs

def get_augmentation_pipeline():
    """
    Constructs a sequential Keras model applying the requested augmentations:
    - Random Rotation
    - Random Flip (Horizontal and Vertical)
    - Random Brightness
    - Random Contrast
    - Random Zoom
    - Random Crop (via Zoom/Crop layers)
    - Random Translation
    - Gaussian Noise
    - Random Saturation
    - Random Hue
    """
    augmentation = tf.keras.Sequential([
        # Random Flip
        layers.RandomFlip("horizontal_and_vertical"),
        
        # Random Rotation
        layers.RandomRotation(factor=0.2, fill_mode="reflect"),
        
        # Random Translation
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="reflect"),
        
        # Random Zoom
        layers.RandomZoom(height_factor=0.1, width_factor=0.1, fill_mode="reflect"),
        
        # Random Brightness & Contrast
        layers.RandomBrightness(factor=0.2),
        layers.RandomContrast(factor=0.2),
        
        # Color Augmentations (Saturation, Hue)
        ColorAugmentationLayer(),
        
        # Gaussian Noise
        GaussianNoiseLayer(stddev=0.03)
    ], name="image_augmentation")
    
    return augmentation
