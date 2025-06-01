from keras import Sequential
from keras.src.layers import Masking, Bidirectional, LSTM, Dropout, Dense
from keras.src.optimizers import Adam
from keras.src.regularizers import regularizers


def build_ernn_model(input_shape_with_timesteps, num_classes=1):
    """
    Construiește modelul ERNN cu LSTM-uri Bidirecționale pentru secvențe de frame-uri.

    Args:
        input_shape_with_timesteps (tuple): Forma inputului (MAX_SEQ_LENGTH, NUM_FEATURES_PER_FRAME).
        num_classes (int): Numărul de clase de ieșire (1 pentru binar cu sigmoid).

    Returns:
        tensorflow.keras.Model: Modelul compilat.
    """
    model = Sequential()

    # Stratul Masking ignoră pașii de timp (frame-urile) care sunt umpluți (păstrați cu 0.0)
    # Este util dacă secvențele au lungimi variabile și sunt umplute cu o valoare specifică.
    model.add(Masking(mask_value=0.0, input_shape=input_shape_with_timesteps))

    model.add(Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=regularizers.L2(0.001))))
    model.add(Dropout(0.4))  # Dropout ajustat

    model.add(Bidirectional(LSTM(32, kernel_regularizer=regularizers.L2(0.001))))  # return_sequences=False by default
    model.add(Dropout(0.4))  # Dropout ajustat

    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.L2(0.001)))

    if num_classes == 1:
        model.add(Dense(1, activation='sigmoid'))
        loss_function = 'binary_crossentropy'
    else:
        model.add(Dense(num_classes, activation='softmax'))
        loss_function = 'categorical_crossentropy'

    model.compile(optimizer=Adam(learning_rate=0.0005),  # Rată de învățare ajustată
                  loss=loss_function,
                  metrics=['accuracy'])
    return model