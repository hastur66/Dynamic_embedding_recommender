import tensorflow as tf
import tensorflow_datasets as tfds
import dataclasses
import functools
from dynamic_embedding.config import max_token_length, punctuation_regex, pad_token, batch_size, seed


def load_datasets():
    # https://www.tensorflow.org/datasets/catalog/movielens
    # Interactions dataset
    raw_ratings_dataset = tfds.load("movielens/1m-ratings", split="train")
    # Candidates dataset
    raw_movies_dataset = tfds.load("movielens/1m-movies", split="train")
    return raw_ratings_dataset, raw_movies_dataset


def process_text(x: tf.Tensor, max_token_length: int, punctuation_regex: str) -> tf.Tensor:

    return tf.strings.split(
        tf.strings.regex_replace(
            tf.strings.lower(x["movie_title"]), punctuation_regex, ""
        )
    )[:max_token_length]


def process_ratings_dataset(ratings_dataset: tf.data.Dataset) -> tf.data.Dataset:

    partial_process_text = functools.partial(
        process_text, max_token_length=max_token_length, punctuation_regex=punctuation_regex
    )

    preprocessed_movie_title_dataset = ratings_dataset.map(
        lambda x: partial_process_text(x)
    )

    processed_dataset = tf.data.Dataset.zip(
        (ratings_dataset, preprocessed_movie_title_dataset)
    ).map(
        lambda x,y: {"user_id": x["user_id"]} | {"movie_title": y}
    )

    return processed_dataset


def process_movies_dataset(movies_dataset: tf.data.Dataset) -> tf.data.Dataset:
    raw_ratings_dataset, raw_movies_dataset = load_datasets()
    partial_process_text = functools.partial(
        process_text, max_token_length=max_token_length, punctuation_regex=punctuation_regex
    )

    processed_dataset = raw_movies_dataset.map(
        lambda x: partial_process_text(x)
    )

    return processed_dataset


def process_and_get_dataset_info(raw_ratings_dataset):
    processed_ratings_dataset = process_ratings_dataset(raw_ratings_dataset, max_token_length, punctuation_regex)
    train_size = int(len(processed_ratings_dataset) * 0.9)
    validation_size = len(processed_ratings_dataset) - train_size
    print(f"Train size: {train_size}")
    print(f"Validation size: {validation_size}")
    return train_size, validation_size


@dataclasses.dataclass(frozen=True)
class TrainingDatasets:
    train_ds: tf.data.Dataset
    validation_ds: tf.data.Dataset

@dataclasses.dataclass(frozen=True)
class RetrievalDatasets:
    training_datasets: TrainingDatasets
    candidate_dataset: tf.data.Dataset


def pad_and_batch_ratings_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:

    return dataset.padded_batch(
        batch_size,
        padded_shapes={
            "user_id": tf.TensorShape([]),
            "movie_title": tf.TensorShape([max_token_length,])
        }, padding_values={
            "user_id": pad_token,
            "movie_title": pad_token
        }
    )


def pad_and_batch_candidate_dataset(movies_dataset: tf.data.Dataset) -> tf.data.Dataset:
    return movies_dataset.padded_batch(
        batch_size,
        padded_shapes=tf.TensorShape([max_token_length,]),
        padding_values=pad_token
    )


def split_train_validation_datasets(ratings_dataset: tf.data.Dataset) -> TrainingDatasets:
    train_size, validation_size = process_and_get_dataset_info(ratings_dataset)
    shuffled_dataset = ratings_dataset.shuffle(buffer_size=5*batch_size, seed=seed)
    train_ds = shuffled_dataset.skip(validation_size).shuffle(buffer_size=10*batch_size).apply(pad_and_batch_ratings_dataset)
    validation_ds = shuffled_dataset.take(validation_size).apply(pad_and_batch_ratings_dataset)

    return TrainingDatasets(train_ds=train_ds, validation_ds=validation_ds)


def create_datasets() -> RetrievalDatasets:

    raw_ratings_dataset = tfds.load("movielens/1m-ratings", split="train")
    raw_movies_dataset = tfds.load("movielens/1m-movies", split="train")

    processed_ratings_dataset = process_ratings_dataset(raw_ratings_dataset)
    processed_movies_dataset = process_movies_dataset(raw_movies_dataset)

    training_datasets = split_train_validation_datasets(processed_ratings_dataset)
    candidate_dataset = pad_and_batch_candidate_dataset(processed_movies_dataset)

    return RetrievalDatasets(training_datasets=training_datasets, candidate_dataset=candidate_dataset)

if __name__ == "__main__":
    train_ds, validation_ds, candidate_dataset = create_datasets()
    print(f"Train dataset size (after batching): {len(train_ds)}")
    print(f"Validation dataset size (after batching): {len(validation_ds)}")
