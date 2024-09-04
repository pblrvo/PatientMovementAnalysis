from tensorflow.keras.layers import Conv1D, BatchNormalization, LSTM, Dense, Dropout
import keras_tuner as kt
import keras

MAX_SEQ_LENGTH = 370

class MyHyperModel(kt.HyperModel):

    def build(self, hp):
        dense_units = hp.Int('dense_units', min_value=64, max_value=512, step=64)
        learning_rate = hp.Float('learning_rate', min_value=1e-7, max_value=1e-3, sampling='LOG')
        dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
        conv1_filters = hp.Int('conv1_filters', min_value=32, max_value=128, step=32)
        conv2_filters = hp.Int('conv2_filters', min_value=64, max_value=256, step=64)
        conv3_filters = hp.Int('conv3_filters', min_value=128, max_value=512, step=128)
        lstm1_units = hp.Int('lstm1_units', min_value=64, max_value=256, step=64)
        lstm2_units = hp.Int('lstm2_units', min_value=32, max_value=128, step=32)

        inputs = keras.Input(shape=(MAX_SEQ_LENGTH, 272))

        x = inputs

        # Conv1D layers with hyperparameter tuning
        x = Conv1D(filters=conv1_filters, kernel_size=3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(conv2_filters, kernel_size=3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(conv3_filters, kernel_size=3, activation='relu')(x)
        x = BatchNormalization()(x)

        # LSTM layers with hyperparameter tuning
        x = LSTM(lstm1_units, return_sequences=True)(x)
        x = Dropout(dropout_rate)(x)
        x = LSTM(lstm2_units, return_sequences=False)(x)
        x = Dropout(dropout_rate)(x)

        x = Dense(dense_units, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(dense_units//2, activation='relu')(x)
        outputs = Dense(4, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        
        # Hyperparameter tuning for optimizers
        optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
        if optimizer_choice == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_choice == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model