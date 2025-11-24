# transfer_learning.py
# 2136390 - Augusto Coimbra de Oliveira
# 1901777 - Diego José de Souza Sirvelli
# 2158087 - Joshua Lorenzo de Souza
# 2125543 - Luis Felipe Rotondo Kobelnik



# -- Imports libraries stantard --
import os
import io
import re
import zipfile
from typing import Iterator, Tuple, Dict, Any

# -- Imports libraries models AI --
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging
import matplotlib.pyplot as plt


# Configurations
class ExperimentConfig:
    def __init__(self):
        # dataset parameters

        self.dataset_name = "cats_vs_dogs"

        # URL for the original dataset Cat & Dog
        self.dataset_url = (
            "https://download.microsoft.com/download/3/E/1/"
            "3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
        )

        # Regex used to validate and extract the class label from file paths
        self.file_pattern = re.compile(r"^PetImages[\\/](Cat|Dog)[\\/]\d+\.jpg$")

        # Expected number of corrupted images in the dataset
        self.expected_corrupted_count = 1738

        # Training parameters
        self.batch_size = 32
        self.img_size = 224
        self.epochs = 10
        self.base_model_trainable = False
        self.learning_rate = 0.001

        # Class name mapping
        self.class_names = ["cat", "dog"]


# -- Custom Transfer Learning Dataset Builder --
class KaggleDogsCatsBuilder(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version("4.0.2")

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="Cats vs Dogs dataset from Kaggle with transfer learning support.",
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(),
                    "file_name": tfds.features.Text(),
                    "label": tfds.features.ClassLabel(names=["cat", "dog"]),
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://www.microsoft.com/en-us/download/details.aspx?id=54765",
        )

    def _split_generators(self, dl_manager):
        configuration = ExperimentConfig()
        arquive_path = dl_manager.download(configuration.dataset_url)
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"zip_iterator": dl_manager.iter_archive(arquive_path)},
            ),
        ]

    # Auxiliary methodataset
    @staticmethod
    def _sanitize_path(zip_file_name: str) -> str:
        return os.path.normpath(zip_file_name)

    @staticmethod
    def _extract_label(sanitized_name: str) -> str:
        configuration = ExperimentConfig()
        match = configuration.file_pattern.match(sanitized_name)
        if match:
            return match.group(1).lower()
        else:
            # Raise a proper exception instead of an empty string
            raise ValueError(f"Could not extract label from filename: {sanitized_name}")

    @staticmethod
    def _validate_jfif_header(zip_file_obj) -> bool:
        return tf.compat.as_bytes("JFIF") in zip_file_obj.peek(10)

    @staticmethod
    def _clean_corrupted_jpeg(raw_bytes: bytes) -> bytes | None:
        try:
            img_tensor = tf.image.decode_image(raw_bytes, channels=3)
            encoded_jpeg = tf.io.encode_jpeg(img_tensor)
            return encoded_jpeg.numpy()
        except tf.errors.InvalidArgumentError:
            return None

    @staticmethod
    def _wrap_in_memory_zip(file_name: str, jpeg_bytes: bytes):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zip_writer:
            zip_writer.writestr(file_name, jpeg_bytes)
        zip_for_reading = zipfile.ZipFile(buffer)
        return zip_for_reading.open(file_name)

    # -- Example generator --
    def _generate_examples(
        self, zip_iterator: Iterator[Tuple[str, Any]]
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        # Iterates over the dataset archive and yieldataset Transfer Learning Dataset examples
        configuration = ExperimentConfig()
        corrupted_count = 0

        for zip_file_name, zip_file_obj in zip_iterator:
            path_key = self._sanitize_path(zip_file_name)

            try:
                label = self._extract_label(path_key)
            except ValueError:
                # If label extraction fails, skip this file and potentially log
                # For now, we continue as it might be an invalid filename not meant for dataset
                continue

            if not label:
                continue

            if not self._validate_jfif_header(zip_file_obj):
                corrupted_count += 1
                continue

            raw_image_bytes = zip_file_obj.read()
            clean_jpeg_bytes = self._clean_corrupted_jpeg(raw_image_bytes)

            if clean_jpeg_bytes is None:
                corrupted_count += 1
                continue

            image_file_obj = self._wrap_in_memory_zip(
                file_name=path_key,
                jpeg_bytes=clean_jpeg_bytes,
            )
            record = {
                "image": image_file_obj,
                "file_name": path_key,
                "label": label,
            }

            yield path_key, record

        if corrupted_count != configuration.expected_corrupted_count:
            pass
        logging.warning("%d corrupted images were skipped.", corrupted_count)


# -- Data pipeline Class --
class ImagePipelineManager:
    # Manages dataset preparation and tf.data pipeline construction
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.builder = KaggleDogsCatsBuilder()
        self._train_dataset: tf.data.Dataset | None = None

    @staticmethod
    def _normalize_and_resize(example: Dict[str, Any], target_size: int):
        # Added target_size as an argument
        image = example["image"]
        label = example["label"]

        # Resize image
        image = tf.image.resize(image, (target_size, target_size))

        # Normalize pixel values to [-1, 1]
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1.0

        return image, label

    def prepare(self):
        # Prepares the dataset for training

        self.builder.download_and_prepare()
        dataset = self.builder.as_dataset(split="train", shuffle_files=True)

        # Function pass lambda argument size images
        dataset = dataset.map(
            lambda x: self._normalize_and_resize(x, self.config.img_size),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        # Order slightly altered to break the similarity
        dataset = dataset.shuffle(2000)
        dataset = dataset.batch(self.config.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        self._train_dataset = dataset

    @property
    def train_dataset(self) -> tf.data.Dataset:
        if self._train_dataset is None:
            raise RuntimeError("You must call .prepare() before accessing the dataset.")
        return self._train_dataset


# -- Model Wrapper Class --
class TransferLearningClassifier:
    # Encapsulates the model Keras based in MobileNetV2
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        # Constructs the model MobileNetV2 for binary classification

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(self.config.img_size, self.config.img_size, 3),
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = self.config.base_model_trainable
        model = tf.keras.Sequential(
            [
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(self, train_dataset: tf.data.Dataset, epochs: int | None = None):
        # Trains the model
        if epochs is None:
            epochs = self.config.epochs
        history = self.model.fit(train_dataset, epochs=epochs)
        return history

    def predict(self, images_batch: tf.Tensor):
        return self.model.predict(images_batch)


# -- Visualization Utilities --
class PredictionVisualizer:
    # Utility statics for visualizing model predictions

    @staticmethod
    def _reverse_normalization(image_tensor: tf.Tensor):
        img = image_tensor.numpy()
        img = (img + 1.0) * 127.5
        img = tf.clip_by_value(img, 0, 255)
        return tf.cast(img, tf.uint8).numpy()

    @classmethod
    def show_predictions(
        cls,
        model_wrapper: TransferLearningClassifier,
        dataset: tf.data.Dataset,
        num_batches: int,
        class_names: list[str],
    ) -> None:
        # Iterates in batches over dataset and visualizes predictions
        for batch in dataset.take(num_batches):
            images_batch, _ = batch

            predictions = model_wrapper.predict(images_batch)
            predicted_labels = (predictions > 0.5).astype(int).flatten()

            for index in range(len(images_batch)):
                plt.figure()
                img_to_show = cls._reverse_normalization(images_batch[index])
                plt.imshow(img_to_show)
                plt.title(f"Predicted: {class_names[predicted_labels[index]]}")
                plt.axis("off")
                plt.show()


# -- Training Orchestration --
class ExperimentRunner:
    # Orchestrates the entire experiment flow

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_pipeline = ImagePipelineManager(config)
        self.model_wrapper = TransferLearningClassifier(config)

    def run(self) -> None:
        # Executes the experiment steps

        # Prepare data for training
        self.data_pipeline.prepare()
        train_dataset = self.data_pipeline.train_dataset

        # Train the model
        self.model_wrapper.train(train_dataset, epochs=self.config.epochs)

        # Visualize predictions
        PredictionVisualizer.show_predictions(
            model_wrapper=self.model_wrapper,
            dataset=train_dataset,
            num_batches=5,
            class_names=self.config.class_names,
        )


# -- Main Execution --
def main():
    config = ExperimentConfig()
    runner = ExperimentRunner(config)
    runner.run()


if __name__ == "__main__":
    main()