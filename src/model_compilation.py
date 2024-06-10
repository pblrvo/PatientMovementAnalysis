import keras
from keras import layers
from data_preparation import build_tensors
import data_collection
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


MAX_SEQ_LENGTH = 3982
EPOCHS = 5

videos_df = data_collection.load_features('./resources/JSON/')
train_data, train_labels, test_data, test_labels = build_tensors(videos_df)
train_data_padded = pad_sequences(train_data, maxlen=MAX_SEQ_LENGTH, padding='post')
test_data_padded = pad_sequences(test_data, maxlen=MAX_SEQ_LENGTH, padding='post')

class TransformerEncoder(keras.layers.Layer):
  def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
    super().__init__(**kwargs)
    self.embed_dim = embed_dim
    self.dense_dim = dense_dim
    self.num_heads = num_heads
    # Self-attention layer for capturing relationships between keypoints within a frame
    self.attention = layers.MultiHeadAttention(
      num_heads=num_heads, key_dim=embed_dim, dropout=0.3
    )
    # Feed-forward layers for non-linear feature extraction
    self.dense_proj = keras.Sequential(
      [
        layers.Dense(dense_dim, activation=keras.activations.gelu),
        layers.Dense(embed_dim),
      ]
    )
    # Layer normalization layers for stabilizing gradients
    self.layernorm_1 = layers.LayerNormalization()
    self.layernorm_2 = layers.LayerNormalization()

  def call(self, inputs, mask=None):
    # Apply self-attention to capture relationships between keypoints
    attention_output = self.attention(inputs, inputs, attention_mask=mask)
    # Add the input to the attention output for residual connection
    proj_input = self.layernorm_1(inputs + attention_output)
    # Apply feed-forward layers for non-linear feature extraction
    ff_output = self.dense_proj(proj_input)
    # Add the feed-forward output to the projected input for residual connection
    return self.layernorm_2(proj_input + ff_output)


def get_compiled_model(shape):
    num_features = 272  
    dense_dim = 4
    num_classes = len(set(train_labels))

    inputs = keras.Input(shape=(MAX_SEQ_LENGTH, num_features))
    x = layers.Masking(mask_value=0)(inputs) 
    # Convolutional layers with BatchNormalization and LeakyReLU
    x = layers.Conv1D(filters=32, kernel_size=3, activation="leaky_relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)  # Added BatchNormalization
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(filters=64, kernel_size=3, activation="leaky_relu", padding="same")(x)
    x = layers.BatchNormalization()(x)  # Added BatchNormalization
    x = layers.Dropout(0.2)(x)

    # Bidirectional LSTMs for capturing temporal dependencies
    x = layers.Bidirectional(layers.LSTM(dense_dim, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(dense_dim))(x)
    x = layers.Dropout(0.2)(x)

    # Final layers for classification
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model



def run_experiment():
    filepath = "/tmp/video_classifier.weights.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)
    # Create a label encoder
    le = LabelEncoder()
    train_labels_encoded = le.fit_transform(train_labels)
    test_labels_encoded = le.transform(test_labels)

    model = get_compiled_model((MAX_SEQ_LENGTH, 272))
    print(model.summary())
    model.fit(
        train_data_padded,
        train_labels_encoded,
        validation_split=0.15,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping],
    )

    model.load_weights(filepath)
    _, accuracy = model.evaluate(test_data_padded, test_labels_encoded)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return model

model = run_experiment()