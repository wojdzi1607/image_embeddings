import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint


# Load pre-trained model
model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(120, 160, 3), pooling="avg")

# Freeze some layers [!]
for layer in model.layers[:-5]:
    layer.trainable = False

batch_size = 16

# Load data
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
valid_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
train_generator_with_data = train_generator.flow_from_directory('data/data_to_train/train', batch_size=batch_size, shuffle=True)
valid_generator_with_data = valid_generator.flow_from_directory('data/data_to_train/val', batch_size=batch_size)

# Add classification layer
trainable_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Dense(58, activation='softmax')
])

# Create checkpoints
checkpoint = ModelCheckpoint(filepath='models/val_model.hdf5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')
callbacks = [checkpoint]

# Compile model
learning_rate = 1e-3
optimizer = Adam(learning_rate)
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

trainable_model.compile(optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

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

# Save model
trainable_model = tf.keras.models.load_model('models/val_model.hdf5.hdf5')

# Remove last layer and save
trainable_model._layers.pop()
trainable_model.save("final_model.hdf5")

