import pandas as pd
from src.task_3.data_segmentation import segment_data

def test_segment_data():
    df = pd.DataFrame({
        'Gender': ['Male', 'Female', 'Male', 'Female'],
        'claim_amount': [100, 200, 150, 250]
    })
    control, test = segment_data(df, 'Gender', 'Female')
    assert all(control['Gender'] != 'Female')
    assert all(test['Gender'] == 'Female')
