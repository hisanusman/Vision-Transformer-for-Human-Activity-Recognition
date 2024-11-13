---

# Human Action Recognition with Vision Transformer (ViT) on HMDB Dataset

This project aims to perform human action recognition using a Vision Transformer (ViT) model fine-tuned on the HMDB (Human Motion Database) dataset. The HMDB dataset includes over 6,800 video clips spanning 51 action categories, such as running, eating, and waving, making it a comprehensive benchmark for human activity recognition. By extracting frames from videos, preprocessing them, and fine-tuning a ViT model, we aim to classify actions accurately with a target accuracy of 90%.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Setup Instructions](#setup-instructions)
4. [Usage](#usage)
5. [Known Issues](#known-issues)
6. [Acknowledgments](#acknowledgments)

---

### Project Overview

The code for this project is divided into the following steps:
1. **Dataset Preprocessing** <br>
 Extracted frames from each video in the HMDB dataset. <br>
 Resized the frames to the input size expected by Vision Transformer (ViT). <br>
 Applied data augmentation techniques to improve generalization, such as cropping, flipping, and normalization.

2. **Loading the Vision Transformer Model** <br>
 Loaded a pre-trained ViT model suitable for image-based tasks. <br>
 Modified the model’s final layers to match the number of classes in the HMDB dataset. <br>
 Used high-level libraries such as hugging face etc.

3. **Setting Up Training Configurations** <br>
 Chose an appropriate batch size and number of epochs for effective training. <br>
 Set a suitable learning rate for fine-tuning the model on the dataset.

4. **Checkpointing and Early Stopping** <br>
 Used checkpointing to save the best-performing model during training. <br>
 Implemented early stopping based on validation performance to avoid overfitting.

5. **Model Evaluation** <br>
 Evaluated the model’s accuracy on the test set and achieved 95% plus accuracy.

---

### Prerequisites

Ensure the following are installed in your environment (these are available in Kaggle notebooks by default):
- Python 3.8+
- PyTorch 1.7+
- Hugging Face’s `transformers` library
- `torchvision`
- `pandas`
- `opencv-python`
- `matplotlib`

```bash
pip install torch torchvision transformers pandas opencv-python matplotlib
```

---

### Setup Instructions

1. **Dataset Preparation**:
   - Upload the HMDB dataset to your Kaggle notebook workspace or local environment. Ensure that the dataset is structured with action class folders containing video frames (as per the project requirements).

2. **Code Structure**:
   - Organize your code cells in the following order to run them sequentially:
     - **Data Preprocessing**: Preprocesses frames for model input.
     - **Model Loading**: Loads and customizes the ViT model.
     - **Training Configuration**: Sets batch size, epochs, learning rate, and optimizers.
     - **Training with Checkpointing and Early Stopping**: Saves the best model.
     - **Evaluation and Visualization**: Evaluates model accuracy and displays sample predictions.

3. **Directory Setup**:
   - Ensure that your dataset is organized under `/kaggle/working/hmdb_frames/` (or adjust `train_data_path` and `val_data_path` in the code to match your setup).

4. **Running the Code**:
   - Execute each section in sequence in a Jupyter notebook environment such as Kaggle notebooks.

---

### Usage

1. **Training the Model**:
   - Run the training script to start model training. Training progress, including losses and accuracies, will be displayed for each epoch.
   - The model will save the best checkpoint based on validation accuracy.

2. **Model Evaluation**:
   - After training, evaluate the model’s performance using the test set.
   - Run the visualization code to plot training/validation losses and visualize predictions for selected test samples.

3. **Model Saving**:
   - The trained model will be saved as `hmdb_vit_model.pth` after training completes.

---

### Known Issues

1. **Dataset Structure**:
   - The HMDB dataset needs to be organized in a specific structure. If the dataset is not structured correctly, the `HMDBDataset` class may throw errors. Ensure frames are extracted into a parent folder with each action class as a sub-folder.

2. **Memory Usage**:
   - Fine-tuning the ViT model is memory-intensive, especially on larger datasets. If running out of memory, consider reducing the batch size, image resolution, or model complexity (e.g., using a smaller ViT variant).

3. **Frame Extraction Requirements**:
   - The `cv2.VideoCapture` function may fail on unsupported video formats. If you encounter issues during frame extraction, check that OpenCV is installed correctly and videos are in a compatible format (e.g., .mp4).

4. **Validation and Test Set Accuracy**:
   - Achieving high validation accuracy (90% or above) on the HMDB dataset can be challenging due to its diversity. Fine-tuning parameters like learning rate, batch size, or data augmentation can help improve results.

5. **Training Speed**:
   - Fine-tuning a ViT model on a large dataset like HMDB may take significant time. Consider using a GPU-based environment for faster training, as provided by Kaggle or Google Colab.

---

### Acknowledgments

- The [HMDB dataset](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) creators for the dataset used in this project.
- Hugging Face for providing the Vision Transformer (ViT) model used in this implementation.

---

This README provides all necessary details for running and troubleshooting the code. Feel free to reach out with any questions or issues related to the code and dataset structure!
