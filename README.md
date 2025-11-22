## Project Details for README

### Project Objective 
The primary objective of this project is to build and train an image classification model to distinguish between images of cats and dogs using transfer learning with a pre-trained MobileNetv2 model.


### Dataset Source and Characteristics
Source UR: `https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip` (Kaggle Cats and Dogs dataset provided by Microsoft).

Custom TFDS Builder: A custom 'CatsVsDogs' 'tfds.core.GeneratorBasedBuilder' is used to load and process the dataset.

Data Processing: The builder extracts images from a Zip arquive, identifiles labels ('cat' or 'dogs') based on directory structure, and includes logic to skip corrupted JPEG images. It specifically checks for 'JFIF' in the file header.

Corrupted Images: The builder expects '1802' corrupted images to be skipped during the dataset generation process.

Features: Each example consists of an 'image' (tfds.features.Image()), 'image/filename' (tfds.features.Text()), and 'label' (tds.features.ClassLabel with names 'cat', 'dog').

### Model Architetura
Base Model: 'tf.keras.applications.MobileNetV2' is used as the base mode.


Transfer Learning: The base model is initialized with 'imagenet' weights.

Input Shape: The input shape for the model is '(224, 224, 3)'.

Include Top: 'include_top = False' is set, meaning the classification head of the MobileNetV2 is not included, allowing for custom classification layers.

Freezing Layers: The 'base_mode.trainable' attribute is set to 'False', effectively freezing the weight of the pre-trained MobileNetV2 base during training to leverage learned features without modifying them.

Custom Classification Head: A 'tf.keras.Sequential' model is built on top of the frozen base model, consisting of:
    'tf.keras.layers.GlobalAveragePooling2D()': Reduces the spatial dimensions of the feature maps to a single feature vector per channel
    'tf.keras.layers.Dense(1, activation = 'sigmoid')': A single dense output layer with a sigmoid activation function for binary classification (cat vs dog).

### Processing Steps

Image Resizing: Images are resized to '(224, 224)' pixels.

Normalization: 'tf.keras.applications.mobilenet_v2.preprocess_input' is applied, wich scales pixel values to the range '[-1, 1]', suitable for MobileNetV2 models.

Batching and Shuffling: The dataset is shuffled with a buffer size of 1000, batched int groups of 32 images, and prefetched for optimized performace

### Training Configuration

Optimizer: 'adam' optimizer

Loss Function: 'binary_crossentropy' (appropriate for binary classification with a sigmoid output)

Metrics: 'accuracy'

Epochs: The model is trained for '5' epochs

Batch Size: '32' (as configured during dataset preparation)

### Expected Results and Visualizations

Training History: The 'history' object from 'model.fit' will contain metrics like loss and accuracy per epoch, which can be visualized to monitor training progress.

Sample Image Predictions: After training, the notebook demonstrates how to make predictions on sample images from the dataset. It visualizes these images along with their predicted labels ('cat' or 'dog') after reversing the MobileNetV2 preprocessing for display purposes. This provides a qualitative assessment of the model's performance on individual images.