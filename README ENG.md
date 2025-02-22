# Model Pruning

## *Dimensionality Reduction of CNNs for Image Processing*

This project was developed as part of the Automation Engineering Project at TU Berlin, within the Department of Industrial Automation Technology. The main goal was to reduce the dimensions of AI models through model pruning, enhancing efficiency and speed while maintaining accuracy as much as possible.

The project resulted in an interface that allows testing of the Taylor pruning method. The pruning process can be applied to CNN models in `.pth` format. A sample ResNet model is available in the `Models` directory.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Ongoing](#ongoing)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## Installation

Before proceeding with the installation, clone this repository:

```bash
# Clone the repository
git clone https://github.com/rabeaifeanyi/CNN-Model-Pruning.git

# Navigate to the project directory
cd CNN-Model-Pruning
```

The app can be installed in three ways: using Docker, Conda, or a virtual environment.

### Installation with Docker

```bash
# Build the Docker image
docker build -t cnn-model-pruning .
```

### Installation with Conda (Recommended)

Refer to the [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) for additional guidance.

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate cnn-model-pruning
```

### Alternative: Manual Installation

Python 3.8 or later is required.

```bash
# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # For Unix or MacOS
venv\Scripts\activate  # For Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Preparation

1. Verify that the installation was successful.
2. Navigate to the *CNN-Model-Pruning* directory.

### Running the Application & Selecting a Model

1. Start the application using one of the following methods:

```bash
# USING DOCKER
# Start the Docker container (if no custom model is available)
docker run --gpus all -p 8080:8080 cnn-model-pruning

# Start the container with a custom model
docker run --gpus all -p 8080:8080 -v <model-path>:/app/Models/<model-name>.pth -d cnn-model-pruning  # Adjust path and model name accordingly

# OTHERWISE
# Start the app using Streamlit
streamlit run main.py
```

2. Open `http://localhost:8080` in a web browser if it does not open automatically. The following interface should appear:

3) Under *Select a model*, choose `Upload from files` and specify the model path:

   - **Docker**  `/Models/<model-name>.pth`  The sample model provided is *resnet\_model\_mnist.pth*.
   - **Otherwise**  Provide the local path to the model.  Example: `<dir-path>/Models/resnet_model_mnist.pth`

4) Specify the dataset path. Example: Select the MNIST dataset.

### Configuring the Pruning Parameters

1. Select the pruning method. *Currently, only the Taylor pruning method is implemented!*
2. Set the number of training epochs.
3. Define the sparsity percentage.
4. If necessary, specify the image size for datasets. *Note: For the sample model, set image size to 224.*
5. Click `Run`.

### Results & Evaluation

1. If successful, the pruning process will start. This may take from a few minutes to several hours.
2. Once completed, the results will be displayed.
3. Pruned models are stored in the *Models\_pruned* directory.

## Ongoing

- [ ] Testing the pipeline with external models
- [ ] Validating installation instructions
- [ ] Making an accessible example project available
- [ ] Automatically detecting models in the `Models` folder instead of requiring manual input
- [x] Completing the user guide
- [ ] Implementing and integrating the APoZ pruning method
- [ ] Investigating why PyTorch pruning methods are not functioning properly

## Future Enhancements

- [ ] Saving intermediate states and pruned models for better comparability (integrate Streamlit session states). Currently, users need to manually input the path of a previously pruned model.
- [ ] Ensuring that computations are not repeated if the process is interrupted or the page is refreshed (use caching or integrate Streamlit session states).

## License

Copyright (c) 2023 TU Berlin, Institute for Machine Tools and Factory Management Department of Industrial Automation Technology Authors: Leandro Carrión Benenwart, Rabea Eschenhagen, Robert Komorowski, Sedat Süzer, Tom Wolf All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions, and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions, and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

### Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

