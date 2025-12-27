"""Downloads and extracts the MNIST dataset."""

import gzip
import os
import struct
import urllib.request

import numpy as np


def load_mnist(data_dir: str = ".datasets"):
    """Load MNIST dataset, downloading if necessary."""
    download_mnist(data_dir)

    # Load training data
    train_images = load_mnist_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    train_labels = load_mnist_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))

    # Load test data
    test_images = load_mnist_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    test_labels = load_mnist_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))

    return train_images, train_labels, test_images, test_labels


def download_mnist(data_dir: str = ".datasets") -> None:
    os.makedirs(data_dir, exist_ok=True)

    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
            print(f"Downloaded {filename}")


def load_mnist_images(filepath: str) -> np.ndarray:
    """Load MNIST images from compressed file."""
    with gzip.open(filepath, "rb") as f:
        # Read header
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Invalid magic number: {magic}"

        # Read image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows * cols)

        # Normalize to [0, 1]
        images = images.astype(np.float32) / 255.0

    return images


def load_mnist_labels(filepath: str) -> np.ndarray:
    """Load MNIST labels from compressed file."""
    with gzip.open(filepath, "rb") as f:
        # Read header
        magic, num_labels = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Invalid magic number: {magic}"

        # Read label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    # Convert to one-hot encoding
    num_classes = 10
    one_hot_labels = np.zeros((len(labels), num_classes), dtype=np.float32)
    one_hot_labels[np.arange(len(labels)), labels] = 1.0
    return one_hot_labels
