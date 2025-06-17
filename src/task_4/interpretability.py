import lime
import lime.lime_tabular
import numpy as np
import pandas as pd

def explain_model_with_lime(model, X_train, X_test, feature_names, instance_idx=0, mode='regression'):
    """
    Explain the prediction of a single instance using LIME.
    
    Parameters:
    - model: trained model (regressor or classifier)
    - X_train: training features as DataFrame or numpy array (for LIME background distribution)
    - X_test: test features as DataFrame or numpy array
    - feature_names: list of feature names (columns)
    - instance_idx: index of the test instance to explain
    - mode: 'regression' or 'classification'
    
    Returns:
    - explanation object from LIME (can be used to display or save explanations)
    """
    if isinstance(X_train, pd.DataFrame):
        training_data = X_train.values
    else:
        training_data = X_train
        
    if isinstance(X_test, pd.DataFrame):
        test_data = X_test.values
    else:
        test_data = X_test
        
    # Setup the LIME explainer for tabular data
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data,
        feature_names=feature_names,
        mode=mode,
        discretize_continuous=True
    )
    
    # Select one instance to explain
    instance = test_data[instance_idx]
    
    # Get explanation
    explanation = explainer.explain_instance(
        instance, 
        model.predict,
        num_features=10  # top 10 features in explanation
    )
    
    return explanation

def show_lime_explanation(explanation):
    """
    Display the LIME explanation in a readable format.
    """
    print("Feature contributions to prediction:")
    for feature, weight in explanation.as_list():
        print(f"{feature}: {weight:.4f}")
