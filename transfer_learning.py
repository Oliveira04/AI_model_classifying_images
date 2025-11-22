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