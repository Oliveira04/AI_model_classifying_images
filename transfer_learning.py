# transfer_learning.py

# -- Imports libraries stantard --
import os
import io
import re
import zipfile
from typing import Iterator, Tuplem, Dict, Any

# -- Imports libraries models AI --
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging
import matplotlib.pyplot as plt

# Configurations
class ExperimentConfig:
    def __init__(self):
        #dataset parameters

        self.dataset_name = 'cats_vs_dogs'

        # URL for the original dataset Cat & Dog
        self.dataset_url = (
        "https://download.microsoft.com/download/3/E/1/"
        "3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip")

        # Regex used to validate and extract the class label from file paths
        self.file_pattern = re.compile(r"^PetImages[\\/](Cat|Dog)[\\/]\d+\.jpg$")
        
        # Expected number of corrupted images in the dataset 
        self.expected_corrupted_count = 1738

        # Training parameters
        self.batch_size = 32
        self.img_size = (224, 224)
        self.epochs = 10
        self.base_model_trainable = False
        self.learning_rate = 0.001

        # Class name mapping
        self.class_names = ['cat', 'dog']

# -- Custom Transfer Learning Dataset Builder --
class KaggleDogsCatsBuilder(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version('4.0.2')

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder = self,
            description = "Cats vs Dogs dataset from Kaggle with transfer learning support.",
            features = tfds.features.FeaturesDict({
                "image": tfds.features.Image(),
                "file_name": tfds.features.Text(),
                "label": tfds.features.ClassLabel(names=['cat', 'dog']),
            }),
            supervised_keys = ("image_raw", "label"),
            homepage="https://www.microsoft.com/en-us/download/details.aspx?id=54765",
        )
    
    def _split_generators(self, dl_manager):
        configuration = ExperimentConfig()
        arquive_path = dl_manager.download_and_extract(configuration.dataset_url)
        return [
            tfds.core.SplitGenerator(
                name = tfds.Split.TRAIN,
                gen_kwargs = {
                    "zip_iterator": dl_manager.iter_arquive(arquive_path)
                },
            ),
        ]
    
    # Auxiliary methods
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
            raise ValueError(f"File path {sanitized_name} does not match expected pattern.")
    
    @staticmethod
    def _validate_jfif_header(zip_file_obj) -> bool:
        return tf.compat.as_bytes("JFIF") in zip_file_obj.peek(10)
    
    @staticmethod
    def _clean_corrupted_jpeg(raw_bytes: bytes) -> bytes | None:
        try:
            img_tensor = tf.image.decode_image(raw_bytes, channels = 3)
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
    