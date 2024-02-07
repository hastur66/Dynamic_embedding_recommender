import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict
from dynamic_embedding.config import pad_token, max_token_length
from data_pipeline import create_datasets

def get_user_id_lookup_layer(dataset: tf.data.Dataset) -> tf.keras.layers.Layer:
    user_lookup_layer = tf.keras.layers.StringLookup(mask_token=None)
    user_lookup_layer.adapt(dataset.map(lambda x: x["user_id"]))
    return user_lookup_layer

def build_user_model(user_id_lookup_layer: tf.keras.layers.StringLookup):
    vocab_size = user_id_lookup_layer.vocabulary_size()
    return tf.keras.Sequential([
        # Fix from https://github.com/keras-team/keras/issues/16101
        tf.keras.layers.InputLayer(input_shape=(), dtype=tf.string),
        user_id_lookup_layer,
        tf.keras.layers.Embedding(vocab_size, 64),
        tf.keras.layers.Dense(64, activation="gelu"),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))

    ], name="user_model")

def get_movie_title_lookup_layer(dataset: tf.data.Dataset) -> tf.keras.layers.Layer:
    movie_title_lookup_layer = tf.keras.layers.StringLookup(mask_token=pad_token)
    movie_title_lookup_layer.adapt(dataset.map(lambda x: x["movie_title"]))
    return movie_title_lookup_layer

def build_item_model(movie_title_lookup_layer: tf.keras.layers.StringLookup):
    vocab_size = movie_title_lookup_layer.vocabulary_size()
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(max_token_length), dtype=tf.string),
        movie_title_lookup_layer,
        tf.keras.layers.Embedding(vocab_size, 64),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation="gelu"),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ], name="item_model")


class TwoTowerModel(tfrs.Model):
  # We derive from a custom base class to help reduce boilerplate. Under the hood,
  # these are still plain Keras Models.

    def __init__(self, user_model: tf.keras.Model, item_model: tf.keras.Model, task: tfrs.tasks.Retrieval):
        super().__init__()

        # Set up user and movie representations.
        self.user_model = user_model
        self.item_model = item_model
        # Set up a retrieval task.
        self.task = task

    def compute_loss(self, features: Dict[str, tf.Tensor], training=False) -> tf.Tensor:
        # Define how the loss is computed.
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.item_model(features["movie_title"])
        return self.task(user_embeddings, movie_embeddings)


def create_two_tower_model(dataset: tf.data.Dataset, candidate_dataset: tf.data.Dataset) -> tf.keras.Model:
    user_id_lookup_layer = get_user_id_lookup_layer(dataset)
    movie_title_lookup_layer = get_movie_title_lookup_layer(dataset)
    user_model = build_user_model(user_id_lookup_layer)
    item_model = build_item_model(movie_title_lookup_layer)
    task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidate_dataset.map(item_model)
        ),
    )

    model = TwoTowerModel(user_model, item_model, task)
    model.compile(optimizer=tf.keras.optimizers.Adam())

    return model


def train_model():
    datasets = create_datasets()

    model = create_two_tower_model(datasets.training_datasets.train_ds, datasets.candidate_dataset)

    """### Training the model"""

    history = model.fit(datasets.training_datasets.train_ds,
                        epochs=10,
                        validation_data=datasets.training_datasets.validation_ds)

    history_standard = history.history
    return model, history_standard


if __name__ == "__main__":
    model, history_standard = train_model()
    model.save("models/basic_model")
    print(history_standard)