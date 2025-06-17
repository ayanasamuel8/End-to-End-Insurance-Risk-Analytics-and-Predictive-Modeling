import pandas as pd

def segment_data(data, feature, test_value):
    """
    Segments the data into two groups:
      - Group A (Control): Plans without the test feature.
      - Group B (Test): Plans with the test feature (i.e., where feature == test_value).
    
    Returns: control, test as two DataFrame objects.
    """
    control = data[data[feature] != test_value].copy()
    test = data[data[feature] == test_value].copy()
    return control, test