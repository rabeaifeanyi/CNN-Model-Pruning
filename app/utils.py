import streamlit as st
import os
import csv

import pruningTaylor
import pruningPyTorch
#import pruningAPoZ


def read_csv(path):
    with open(path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            metric = row['Metric']
            value = float(row['Value'])
            
            if metric == 'Accuracy Original Model':
                accuracy_original = value
            elif metric == 'Accuracy Randomly Pruned Model':
                accuracy_random = value
            elif metric == 'Accuracy Magnitude Pruned Model':
                accuracy_magnitude = value
            elif metric == 'Mean F1 Score Random Pruning':
                mean_f1_score_random = value
            elif metric == 'Mean F1 Score Magnitude Pruning':
                mean_f1_score_magnitude = value
            elif metric == 'Standard Deviation of F1 Score Random Pruning':
                std_f1_score_random = value
            elif metric == 'Standard Deviation of F1 Score Magnitude Pruning':
                std_f1_score_magnitude = value
            elif metric == 'Mean Random Pruned Training Time':
                mean_time_random_training = value
            elif metric == 'Mean Magnitude Pruned Training Time':
                mean_time_magnitude_training = value

    # Create the results tuple
    results = (
        accuracy_original,
        accuracy_random,
        accuracy_magnitude,
        mean_f1_score_random,
        mean_f1_score_magnitude,
        std_f1_score_random,
        std_f1_score_magnitude,
        mean_time_random_training,
        mean_time_magnitude_training
    )
    
    return results


def calculate_evaluation(selected_methods, model_path, dataset, dataset_path, code):
    """
    
    """
    results = {}

    if "Taylor" in selected_methods:
        if not os.path.exists(f'tylorResults_{code}.csv'):
            st.markdown(":red[Running Taylor Pruning Methods...]")
            # pruningTaylor.run(model_path, dataset, dataset_path, code)
        results["Taylor"] = read_csv(f'pyTorchResults{code}.csv')

    if "APoZ" in selected_methods:
        if not os.path.exists(f'tylorResults_{code}.csv'):
            st.markdown(":red[Running APoZ Pruning Methods...]")
            # p_APoZ.run(model_path, dataset, dataset_path, code)
        

    if "PyTorch" in selected_methods:
        if not os.path.exists(f'pyTorchResults_{code}.csv'):
            st.markdown(":red[Running PyTorch Pruning Methods...]")
            pruningPyTorch.run(model_path, dataset, dataset_path, code)

        results["PyTorch"] = read_csv(f'pyTorchResults{code}.csv')

    return results


def create_evaluation(results):
    """ 
    accuracy_original,
    accuracy_random,
    accuracy_magnitude,
    mean_f1_score_random,
    mean_f1_score_magnitude,
    std_f1_score_random,
    std_f1_score_magnitude,
    mean_time_random_training,
    mean_time_magnitude_training
    """

    st.markdown("<h3 style='color: #708090;'>Evaluation</h3>", unsafe_allow_html=True)

    for result in results:
        if result == "PyTorch":
            accuracy_original, mean_accuracy_random, mean_accuracy_magnitude, mean_f1_score_random, mean_f1_score_magnitude, std_f1_score_random, std_f1_score_magnitude, mean_time_random_training, mean_time_magnitude_training = results[result]

            st.markdown("\n**Cross-Validation Results**")
            st.text("Mean F1 Score Random Pruning: {:.4f}".format(mean_f1_score_random))
            st.text("Standard Deviation of F1 Score Random Pruning: {:.4f}".format(std_f1_score_random))
            st.text("Mean Accuracy Random Pruning: {:.4f}".format(mean_accuracy_random))
            st.text("Mean Random Pruned Training Time: {:.4f}".format(mean_time_random_training))

            st.text("Mean F1 Score Magnitude Pruning: {:.4f}".format(mean_f1_score_magnitude))
            st.text("Standard Deviation of F1 Score Magnitude Pruning: {:.4f}".format(std_f1_score_magnitude))
            st.text("Mean Accuracy Magnitude Pruning: {:.4f}".format(mean_accuracy_magnitude))
            st.text("Mean Magnitude Pruned Training Time: {:.4f}".format(mean_time_magnitude_training))
            
        if result == "Taylor":
            pass
        
        if result == "APoZ":
            pass
        