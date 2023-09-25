import datetime

import streamlit as st
from streamlit_extras.stoggle import stoggle
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.switch_page_button import switch_page

from utils import *
import infos


# Retrieve date for file export
today = datetime.datetime.today().strftime('%Y%m%d')


# Setup of page
st.set_page_config(page_title='Model Pruning', layout='wide', initial_sidebar_state='collapsed')


# Initializing parameters
models = {"Choose a Neural Network": None,
          "Upload from files": None,
          "Basic CNN with MNIST dataset": "Models/model.pth"}

datasets = {"Choose a Dataset": None,
            "Upload from files": None,
            "MNIST": "Datasets/MNIST"}

methods = {"Taylor": "Taylor Expansion-based Filter Pruning",
           "APoZ": "APoZ Pruning",
           "PyTorch": "PyTorch Pruning Methods (random and magnitude)"}

default_model = "Choose a Neural Network"
default_dataset = "Choose a Dataset"


# Eine Lösung für die Sessions finden
code = ""


# Setting the default variables
selected_model = st.session_state.get("selected_model", default_model)
selected_dataset = st.session_state.get("selected_dataset", default_dataset)
selected_methods = st.session_state.get("selected_methods", [])
model_upload_option_selected = st.session_state.get("model_upload_option_selected", False)
dataset_upload_option_selected = st.session_state.get("dataset_upload_option_selected", False)


# New Session State
if 'calculated_configs' not in st.session_state:
    st.session_state.calculated_configs = 0


# Implementation of Introduction part 
with st.container():
    st.title(infos.title)
    st.markdown(infos.welcome)
    stoggle("More information.", infos.more_infos)
    st.markdown("---")


# Implementation of Configuration part 
with st.container():
    st.markdown(f"<h3 style='color: #708090;'>{infos.config_title}</h3>", unsafe_allow_html=True)
    st.subheader(infos.config_subheader)

    selected_model = st.selectbox("Select a model", models, key="model_selection")

    if selected_model == "Basic CNN with MNIST dataset":
        selected_dataset = "MNIST"
        
    if selected_model == "Upload from files":
        model_upload_option_selected = True 

    if model_upload_option_selected:  
        model_path = st.text_input("Please enter path to model")
        if model_path:
            st.text(f"You selected {model_path}")
            models["Upload from files"] = model_path

        selected_dataset = st.selectbox("Select a dataset", datasets, key="dataset_selection")

        if selected_dataset == "Upload from files":
            dataset_upload_option_selected = True

        if dataset_upload_option_selected:
            dataset_path = st.text_input("Please enter path to dataset")

            if dataset_path:
                st.text(f"You selected {dataset_path}")
                datasets["Upload from files"] = dataset_path

    add_vertical_space(1)

    st.subheader("Available Pruning methods")

    for method in methods:
        if st.checkbox(methods[method], value=method in selected_methods):
            if method in selected_methods:
                selected_methods.remove(method)
            else:
                selected_methods.append(method)

if selected_model != "Choose a Neural Network" and selected_methods:
    if st.button("Run"):
        st.session_state.calculated_configs += 1
        st.markdown("---")
        
        with st.container():
            results = calculate_evaluation(selected_methods, models[selected_model], selected_dataset, datasets[selected_dataset], code)
            st.empty() # funktioniert nicht!
        
        create_evaluation(results)



