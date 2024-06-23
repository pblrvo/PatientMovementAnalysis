from keras import layers
import keras_tuner as kt
import keras

MAX_SEQ_LENGTH = 1000
EPOCHS = 200
DENSE_DIM = 272
NUM_CLASSES = 4

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        num_heads = hp.Int('num_heads', min_value=2, max_value=8, step=2)
        num_layers = hp.Int('num_layers', min_value=1, max_value=6, step=1)
        dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
        ff_dim = hp.Int('ff_dim', min_value=32, max_value=256, step=32)
        lstm_units = hp.Int('lstm_units', min_value=64, max_value=256, step=64)
        dense_units = hp.Int('dense_units', min_value=128, max_value=512, step=128)
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')
        
        inputs = keras.Input(shape=(MAX_SEQ_LENGTH, DENSE_DIM))
        
        x = inputs
        
        for _ in range(num_layers):
            x = transformer_encoder(x, head_size=DENSE_DIM, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout_rate)

        # LSTM layers
        x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Bidirectional(layers.LSTM(lstm_units))(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Additional Dense layers
        x = layers.Dense(dense_units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(dense_units // 2, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        
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
