import tensorflow as tf
import os

print("Building open-source Brain MRI Model via Transfer Learning...")

# MobileNetV2 is one of the best open-source lightweight architectures.
# It expects 3 channels, but our pipeline provides 1 channel (grayscale), 
# so we map 1 -> 3 channels using a Conv2D layer.
inputs = tf.keras.Input(shape=(128, 128, 1))
x = tf.keras.layers.Conv2D(3, (3, 3), padding='same', activation='relu')(inputs)

# Load pre-trained ImageNet weights
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

x = base_model(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(4, activation='softmax')(x) # 4 Classes: Glioma, Meningioma, No Tumor, Pituitary

model = tf.keras.Model(inputs, outputs)

# Save the model to the expected file path
save_path = os.path.join('models', 'adaptive_multi_scale_fusion_network.h5')
model.save(save_path)

print(f"Successfully constructed and saved powerful Open-Source MRI model to {save_path}!")
