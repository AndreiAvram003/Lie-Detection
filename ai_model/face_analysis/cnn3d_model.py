from keras import Sequential
from keras.src.layers import Conv3D, BatchNormalization, MaxPooling3D, Flatten, Dense, Dropout

def create_3d_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape, padding='same'
              ),
        BatchNormalization(),
        MaxPooling3D(pool_size=(1, 2, 2), padding='same'),

        Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'
              ),
        BatchNormalization(),
        MaxPooling3D(pool_size=(2, 2, 2), padding='same'),

        Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'
              ),
        BatchNormalization(),
        MaxPooling3D(pool_size=(2, 2, 2), padding='same'),

        Flatten(),
        Dense(256, activation='relu'
             ),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model