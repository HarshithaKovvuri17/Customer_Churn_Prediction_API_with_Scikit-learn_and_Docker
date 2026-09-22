import os
import pytest
import pandas as pd
from app.model import load_data, build_pipeline, train_and_save_model, load_model_pipeline
from app.utils import clean_data, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN
