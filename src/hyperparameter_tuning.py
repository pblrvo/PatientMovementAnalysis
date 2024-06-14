from keras import layers
import keras_tuner as kt
import keras

MAX_SEQ_LENGTH = 600
EPOCHS = 100
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
        num_layers = hp.Int('num_layers', min_value=2, max_value=4, step=1)
        dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
        ff_dim = hp.Int('ff_dim', min_value=32, max_value=128, step=32)
        
        inputs = keras.Input(shape=(MAX_SEQ_LENGTH, DENSE_DIM))
        #x = layers.Masking(mask_value=0)(inputs)
        
        for _ in range(num_layers):
            x = transformer_encoder(inputs, head_size=DENSE_DIM, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout_rate)

        # LSTM layers
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Bidirectional(layers.LSTM(128))(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Hyperparameter tuning for optimizers
        optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model