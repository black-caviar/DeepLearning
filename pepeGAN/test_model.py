import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
import load_dataset as ld
import sys

b = ld.get_image_list(sys.argv[1])
w = ld.get_image_list(sys.argv[2])
data = ld.construct_datasets(w,b)
data = data.map(lambda x,y: (ld.load_256(x),y))
data = data.map(ld.mobilenet_preprocess)
ld.show_dataset(data)


model = keras.Sequential([
    #layers.Input((256,256,3)),
    layers.Input((224,224,3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
data = data.batch(10)
epochs=10
history = model.fit(
    data,
    epochs=epochs
)


