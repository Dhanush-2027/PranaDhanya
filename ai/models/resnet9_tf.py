import tensorflow as tf
from tensorflow.keras import layers, Model

def basic_block(x, out_channels, stride=1):
    """
    Residual block with skip connections.
    Implements: Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN -> (+ skip) -> ReLU
    """
    in_channels = x.shape[-1]
    
    # Main path
    out = layers.Conv2D(
        out_channels, 
        kernel_size=3, 
        strides=stride, 
        padding='same', 
        use_bias=False,
        kernel_initializer='he_normal'
    )(x)
    out = layers.BatchNormalization()(out)
    out = layers.Activation('relu')(out)
    
    out = layers.Conv2D(
        out_channels, 
        kernel_size=3, 
        strides=1, 
        padding='same', 
        use_bias=False,
        kernel_initializer='he_normal'
    )(out)
    out = layers.BatchNormalization()(out)
    
    # Shortcut path
    if in_channels == out_channels and stride == 1:
        shortcut = x
    else:
        shortcut = layers.Conv2D(
            out_channels, 
            kernel_size=1, 
            strides=stride, 
            padding='same', 
            use_bias=False,
            kernel_initializer='he_normal'
        )(x)
        shortcut = layers.BatchNormalization()(shortcut)
        
    out = layers.add([out, shortcut])
    out = layers.Activation('relu')(out)
    return out

def ResNet9TF(input_shape=(224, 224, 3), num_classes=10, augmentation_pipeline=None):
    """
    Constructs ResNet9 model in TensorFlow / Keras.
    """
    inputs = layers.Input(shape=input_shape)
    
    x = inputs
    # If training data augmentation is passed, it is applied here
    if augmentation_pipeline is not None:
        x = augmentation_pipeline(x)
        
    # Stem
    x = layers.Conv2D(
        64, 
        kernel_size=3, 
        strides=1, 
        padding='same', 
        use_bias=False,
        kernel_initializer='he_normal'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Stage 1
    x = basic_block(x, 64, stride=1)
    x = layers.SpatialDropout2D(0.2)(x)
    
    # Stage 2
    x = layers.Conv2D(
        128, 
        kernel_size=3, 
        strides=2, 
        padding='same', 
        use_bias=False,
        kernel_initializer='he_normal'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = basic_block(x, 128, stride=1)
    x = layers.SpatialDropout2D(0.2)(x)
    
    # Stage 3
    x = layers.Conv2D(
        256, 
        kernel_size=3, 
        strides=2, 
        padding='same', 
        use_bias=False,
        kernel_initializer='he_normal'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = basic_block(x, 256, stride=1)
    x = layers.SpatialDropout2D(0.2)(x)
    
    # Stage 4
    x = layers.Conv2D(
        512, 
        kernel_size=3, 
        strides=2, 
        padding='same', 
        use_bias=False,
        kernel_initializer='he_normal'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = basic_block(x, 512, stride=1)
    x = layers.SpatialDropout2D(0.2)(x)
    
    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(
        num_classes, 
        activation='softmax',
        kernel_initializer='he_normal'
    )(x)
    
    model = Model(inputs, outputs, name="resnet9_tf")
    return model
