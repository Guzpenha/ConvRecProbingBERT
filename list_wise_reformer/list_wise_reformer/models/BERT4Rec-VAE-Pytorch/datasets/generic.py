from .base import AbstractDataset

import pandas as pd

from datetime import date


class PreprocessedDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'preprocessed'

    @classmethod
    def url(cls):
        return ''

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return []

    def load_ratings_df(self):
        return []


