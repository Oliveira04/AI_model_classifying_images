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

