# MarineSIR: Marine Spatiotemporal Image Reconstruction

MarineSIR is a high-performance application designed for the reconstruction of missing data in marine satellite imagery, such as Chlorophyll-a concentrations. It leverages a sophisticated deep learning model, the Fourier Transform Convolutional Long Short-Term Memory (FTC-LSTM) network, to accurately fill gaps caused by cloud cover or sensor malfunction.

## Architecture

The MarineSIR system is engineered with a modular architecture to streamline the image reconstruction workflow, from initial data ingestion to final analysis. The system is composed of three primary stages:

1.  **Data Preprocessing Pipeline**: This stage prepares raw satellite data for the model. It includes automated processes for handling missing values (NaNs) and applying feature engineering transformations like Log transforms, standardization, and normalization to optimize data distribution for model training.

2.  **Core Reconstruction Engine (FTC-LSTM)**: The heart of MarineSIR is the FTC-LSTM network, an encoder-decoder architecture proven to capture complex spatiotemporal patterns. Its key innovation is the FTC block, which uses a Fast Fourier Transform to operate in the frequency domain, giving the model a global receptive field to effectively model long-range spatial dependencies like ocean currents and gyres.

3.  **Application Workflow (GUI)**: A user-friendly Graphical User Interface (GUI) allows users to apply pre-trained models for inference. This integrated design supports both research (training new models) and operational use cases (applying existing models for immediate data reconstruction).

## Features

-   **Advanced Deep Learning Model**: Utilizes an FTC-LSTM network for state-of-the-art spatiotemporal reconstruction.
-   **Interactive GUI**: An intuitive interface for running reconstruction, configuring settings, and visualizing results.
-   **Dockable Panels**: Customize your workspace with panels for controls, validation metrics, time-series plots, and logs.
-   **Data Visualization**: Includes an interactive map display for reconstructed data and linked time-series plots for specific coordinates.
-   **Model Training & Inference**: Supports both training new models from scratch (`train.py`) and applying pre-trained models via the GUI (`main.py`).

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/siyuChen540/MarineSIR.git
    cd marinesir
    ```

2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the GUI Application

To launch the MarineSIR application for data reconstruction using a pre-trained model:

```bash
python main.py
```

### Training a New Model
To train the FTC-LSTM model on your own dataset:
1.  Configure the training parameters in config.yml.
2.  Run the training script:
    ```bash
    python train.py
    ```

## Evaluation Metrics
The model's performance is assessed using a comprehensive suite of four metrics:
-   **Root Mean Square Error (RMSE)**
-   **Coefficient of Determination (R²)**
-   **Peak Signal-to-Noise Ratio (PSNR)**
-   **Structural Similarity Index (SSIM)**
