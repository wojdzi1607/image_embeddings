import tensorflow as tf
import warnings

from datetime import datetime
from tensorflow.keras.optimizers import Adam
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

warnings.filterwarnings("ignore")


# Load pre-trained model
# model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(120, 160, 3), pooling="avg")
model = MobileNet(weights="imagenet", include_top=False, input_shape=(120, 160, 3), pooling="avg")

print(model.summary())
# Freeze some layers [!]
for layer in model.layers[:-23]:
    layer.trainable = False
    print(layer.name)

# for layer in model.layers:
#     if layer.name[-2:] == 'bn':
#         layer.trainable = False

print(model.summary())
batch_size = 32

# Load data
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255,
                                                                  rotation_range=15,
                                                                  width_shift_range=5,
                                                                  height_shift_range=5)
valid_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
train_generator_with_data = train_generator.flow_from_directory('data/data_to_train/data_to_train_nop/train', batch_size=batch_size, shuffle=True)
valid_generator_with_data = valid_generator.flow_from_directory('data/data_to_train/data_to_train_nop/val', batch_size=batch_size)

# Add classification layer
trainable_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Dense(57, activation='softmax')
])

# Create callbacks
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(log_dir=logdir)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=4)
checkpoint = ModelCheckpoint(filepath='models/val_model.hdf5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')
earlystopping = EarlyStopping(monitor='val_loss', patience=5)
callbacks = [checkpoint, earlystopping, reduce_lr]

# Compile model
learning_rate = 1e-3
optimizer = Adam(learning_rate)
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

trainable_model.compile(optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
# metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
training_samples = train_generator_with_data.n
validation_samples = valid_generator_with_data.n

print(trainable_model.summary())

# Train model
trainable_model.fit(
    train_generator_with_data,
    steps_per_epoch=training_samples // batch_size,
    validation_data=valid_generator_with_data,
    validation_steps=validation_samples // batch_size,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)

# Remove last layer and save best model
trainable_model = tf.keras.models.load_model('models/val_model.hdf5')
trainable_model._layers.pop()
trainable_model.save('models/final_model.hdf5')
