import tensorflow as tf
from keras import Sequential
from keras.src.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout


def create_2d_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'
               # , kernel_regularizer=regularizers.l2(0.001) # Exemplu L2
              ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), padding='same'),

        Conv2D(64, (3, 3), activation='relu', padding='same'
               # , kernel_regularizer=regularizers.l2(0.001) # Exemplu L2
              ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), padding='same'),

        Conv2D(128, (3, 3), activation='relu', padding='same'
               # , kernel_regularizer=regularizers.l2(0.001) # Exemplu L2
              ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), padding='same'),

        Flatten(),
        Dense(256, activation='relu'
              # , kernel_regularizer=regularizers.l2(0.001) # Exemplu L2
             ),
        BatchNormalization(),
        Dropout(0.5),  # <-- MODIFICARE: Strat Dropout adăugat
        Dense(num_classes, activation='softmax')
    ])

    return model
