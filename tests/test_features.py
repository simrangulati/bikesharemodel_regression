
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from bikeshare_model.config.core import config
from bikeshare_model.processing.data_manager import get_year_and_month
from bikeshare_model.processing.features import *

# def test_get_year_and_month(sample_input_data):
#     print(sample_input_data.head(2))
#     df = get_year_and_month(sample_input_data)
#     assert np.isnan(df['yr'])
#     # assert np.isnan(df['yr'])
#     assert np.isnan(df['mnth'])

def test_WeekdayOneHotEncoder(sample_input_data):
    print(f"shape of x test---{sample_input_data.shape}")
    transformer = WeekdayOneHotEncoder(config.model_config.weekday_var)
    n_cols =  len(sample_input_data.columns)
    print(n_cols)
    subject = transformer.fit(sample_input_data).transform(sample_input_data)
    assert len(subject.columns)==n_cols+8

