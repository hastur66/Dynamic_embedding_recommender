import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_recommenders_addons.dynamic_embedding as de
from typing import Dict
from dynamic_embedding import max_token_length
from data_pipeline import create_datasets
from basic_model import get_user_id_lookup_layer, get_movie_title_lookup_layer

tf.compat.v1.reset_default_graph()

def build_de_user_model(user_id_lookup_layer: tf.keras.layers.StringLookup) -> tf.keras.layers.Layer:
    vocab_size = user_id_lookup_layer.vocabulary_size()
    return tf.keras.Sequential([
        # Fix from https://github.com/keras-team/keras/issues/16101
        tf.keras.layers.InputLayer(input_shape=(), dtype=tf.string),
        user_id_lookup_layer,
        de.keras.layers.Embedding(
            embedding_size=64,
            initializer=tf.random_uniform_initializer(),
            init_capacity=int(vocab_size*0.8),
            restrict_policy=de.FrequencyRestrictPolicy,
            name="UserDynamicEmbeddingLayer"
        ),
        tf.keras.layers.Dense(64, activation="gelu"),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))

    ], name='user_model')

def build_de_item_model(movie_title_lookup_layer: tf.keras.layers.StringLookup) -> tf.keras.layers.Layer:
    vocab_size = movie_title_lookup_layer.vocabulary_size()
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(max_token_length), dtype=tf.string),
        movie_title_lookup_layer,
        de.keras.layers.SquashedEmbedding(
            embedding_size=64,
            initializer=tf.random_uniform_initializer(),
            init_capacity=int(vocab_size*0.8),
            restrict_policy=de.FrequencyRestrictPolicy,
            combiner="mean",
            name="ItemDynamicEmbeddingLayer"
        ),
        tf.keras.layers.Dense(64, activation="gelu"),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ])

"""### Defining the callback to control and log dynamic embeddings"""

class DynamicEmbeddingCallback(tf.keras.callbacks.Callback):

    def __init__(self, model, steps_per_logging, steps_per_restrict=None, restrict=False):
        self.model = model
        self.steps_per_logging = steps_per_logging
        self.steps_per_restrict = steps_per_restrict
        self.restrict = restrict

    def on_train_begin(self, logs=None):
        self.model.dynamic_embedding_history = {}

    def on_train_batch_end(self, batch, logs=None):

        if self.restrict and self.steps_per_restrict and (batch+1) % self.steps_per_restrict == 0:

            [
                self.model.embedding_layers[k].params.restrict(
                    int(self.model.lookup_vocab_sizes[k]*0.8),
                    trigger=self.model.lookup_vocab_sizes[k]-2 # UNK & PAD tokens
                ) for k in self.model.embedding_layers.keys()
            ]

        if (batch+1) % self.steps_per_logging == 0:

            embedding_size_dict = {
                k:self.model.embedding_layers[k].params.size().numpy()
                for k in self.model.embedding_layers.keys()
            }

            for k, v in embedding_size_dict.items():
                self.model.dynamic_embedding_history.setdefault(f"embedding_size_{k}", []).append(v)
            self.model.dynamic_embedding_history.setdefault(f"step", []).append(batch+1)

"""## Defining two tower model with dynamic embeddings"""

class DynamicEmbeddingTwoTowerModel(tfrs.Model):
  # We derive from a custom base class to help reduce boilerplate. Under the hood,
  # these are still plain Keras Models.

    def __init__(self,user_model: tf.keras.Model,item_model: tf.keras.Model,task: tfrs.tasks.Retrieval):
        super().__init__()

        # Set up user and movie representations.
        self.user_model = user_model
        self.item_model = item_model

        self.embedding_layers = {
            "user": user_model.layers[1],
            "movie": item_model.layers[1]
        }

        if not all(["embedding" in v.name.lower() for k,v in self.embedding_layers.items()]):
            raise TypeError(
                f"""All layers in embedding_layers must be embedding layers.
                Got {[v.name for v in self.embedding_layers.values()]}"""
            )

        self.lookup_vocab_sizes = {
            "user": user_model.layers[0].vocabulary_size(),
            "movie": item_model.layers[0].vocabulary_size()
        }
        self.dynamic_embedding_history = {}
        # Set up a retrieval task.
        self.task = task

    def compute_loss(self, features: Dict[str, tf.Tensor], training=False) -> tf.Tensor:
        # Define how the loss is computed.
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.item_model(features["movie_title"])
        return self.task(user_embeddings, movie_embeddings)

def create_de_two_tower_model(dataset: tf.data.Dataset, candidate_dataset: tf.data.Dataset) -> tf.keras.Model:

    user_id_lookup_layer = get_user_id_lookup_layer(dataset)
    movie_title_lookup_layer = get_movie_title_lookup_layer(dataset)
    user_model = build_de_user_model(user_id_lookup_layer)
    item_model = build_de_item_model(movie_title_lookup_layer)
    task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidate_dataset.map(item_model)
        ),
    )

    model = DynamicEmbeddingTwoTowerModel(user_model, item_model, task)
    optimizer = de.DynamicEmbeddingOptimizer(tf.keras.optimizers.Adam())
    model.compile(optimizer=optimizer)

    return model

datasets = create_datasets()
de_model = create_de_two_tower_model(datasets.training_datasets.train_ds, datasets.candidate_dataset)

"""### Training the model with dynamic embeddings

"""

epochs = 10

history_de = {}
history_de_size = {}
de_callback = DynamicEmbeddingCallback(de_model, steps_per_logging=20)

for epoch in range(epochs):

    datasets = create_datasets()
    train_steps = len(datasets.training_datasets.train_ds)

    hist = de_model.fit(
        datasets.training_datasets.train_ds,
        epochs=1,
        validation_data=datasets.training_datasets.validation_ds,
        callbacks=[de_callback]
    )

    for k,v in de_model.dynamic_embedding_history.items():
        if k=="step":
            v = [vv+(epoch*train_steps) for vv in v]
        history_de_size.setdefault(k, []).extend(v)

    for k,v in hist.history.items():
        history_de.setdefault(k, []).extend(v)

