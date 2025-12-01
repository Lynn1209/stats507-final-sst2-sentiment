# src/data.py
from datasets import load_dataset
from typing import Dict


def load_sst2():
    """
    Load the GLUE SST-2 dataset from Hugging Face Datasets.

    Returns
    -------
    dataset : datasets.DatasetDict
        A dict with 'train', 'validation', and 'test' splits.
    """
    dataset = load_dataset("glue", "sst2")
    return dataset


def get_label_names() -> Dict[int, str]:
    """
    Label mapping for SST-2.
    0 -> negative
    1 -> positive
    """
    return {0: "negative", 1: "positive"}
