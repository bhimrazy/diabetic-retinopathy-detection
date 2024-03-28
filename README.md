---
title: Diabetic Retinopathy Detection App
emoji: 🐢
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.22.0
app_file: app.py
license: mit
---


![diabetic-retinopathy-detection](https://github.com/bhimrazy/diabetic-retinopathy-detection/assets/46085301/bb45b4cf-9441-435f-819a-176226e1ac00)


# Diabetic Retinopathy Detection: Utilizing Multiprocessing for Processing Large Datasets and Transfer Learning to Fine-Tune Deep Learning Models
Efficiently process large datasets & develop advanced model pipelines for diabetic retinopathy detection. Streamlining diagnosis.

## TL;DR: 
In this project, large datasets are efficiently handled by downloading, extracting, and preparing them for analysis. Utilizing PyTorch Lightning, a robust system for diabetic retinopathy detection is developed, categorizing images into distinct disease stages. The model pipeline is enriched with various pretrained backbone models, with progress tracked using TensorBoard. Furthermore, a user-friendly web app is created to showcase the model's capabilities. The approach pursued aims to streamline both data processing and model development, facilitating accurate and accessible diabetic retinopathy diagnosis.

## Getting Started
**Introduction:**
Diabetic retinopathy (DR) remains a significant global health concern, with early detection playing a critical role in preventing vision loss. For those eager to contribute to this vital area of research, a comprehensive project studio is readily available. This studio has already tackled many essential tasks involved in DR detection, providing researchers and enthusiasts with a ready-to-use platform for experimentation.

**Get Started with the Project Studio:**
Researchers and enthusiasts alike can access the necessary tools and resources by duplicating this project studio. This streamlined solution offers an immediate starting point for experimentation on the [Diabetic Retinopathy Dataset](https://www.kaggle.com/c/diabetic-retinopathy-detection).

**What the Studio Offers:**
- Efficient Handling of Large Datasets: The studio automates the management of large datasets, including downloading, extracting, and data preparation.
- Advanced Model Development: Utilizing PyTorch Lightning, the studio facilitates the development of a sophisticated system for DR detection, categorizing images into different disease stages.
- Integration of Pretrained Backbone Models: Various pretrained backbone models are integrated into the pipeline, allowing for experimentation with different architectures.
- Progress Tracking with TensorBoard: Researchers can monitor progress seamlessly with TensorBoard integration, tracking metrics and visualizing model performance.
- User-Friendly Web Application: A user-friendly web application is provided for showcasing model capabilities and sharing findings effortlessly.


Here's a more structured and standardized version of the steps in a blog format:

---

## Downloading and Preprocessing Diabetic Retinopathy Dataset:

> Note: You can skip this entire step, as this studio already has it done for you.

In this step, we'll walk through the process of downloading and preprocessing the Diabetic Retinopathy Detection dataset. This dataset is commonly used for developing algorithms to identify diabetic retinopathy in eye images.

### Prerequisites

Before we begin, ensure you have the following prerequisites:

- Kaggle API key (Get one [here](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2Faccount%2Flogin%3Fphase%3Dregister))
- `kaggle` library installed (`pip install kaggle`)

**Note:** Before proceeding with the steps below, make sure to change your current directory to `dr-detection` and install the required dependencies by running the following commands:
```bash
cd dr-detection
pip install -r requirements.txt
```

### Step 1: Download the Dataset

There are two ways to download the dataset:

#### First Way: Downloading as a Complete Zip File

```bash
kaggle competitions download -c diabetic-retinopathy-detection

# Extract
unzip diabetic-retinopathy-detection.zip -d data/diabetic-retinopathy-detection
rm diabetic-retinopathy-detection.zip
```

#### Second Way: Downloading as Parts

```bash
./scripts/download-dr-dataset.sh

# Merge and extract the parts
./scripts/merge_and_extract.sh
```

### Step 2: Preprocess Images

Once the dataset is downloaded, preprocess the images to crop and resize them.

<div align="center">
  <img src="https://github.com/bhimrazy/diabetic-retinopathy-detection/assets/46085301/9fa28dea-38cd-4fba-abb0-0ed8001a8075" alt="Preprocessing Image" height="400" width="auto">
  <p style="text-align:center;">Example of cropping and resizing</p>
</div>

```bash
python scripts/crop_and_resize.py --src data/diabetic-retinopathy-dataset/train data/diabetic-retinopathy-dataset/resized/train
python scripts/crop_and_resize.py --src data/diabetic-retinopathy-dataset/test data/diabetic-retinopathy-dataset/resized/test
```

### Step 3: Split Data and Save to CSV

Finally, split the data into train and validation sets and save them to CSV files.

```bash
python scripts/split_dataset.py
```

## Training Model and Monitoring Progress with TensorBoard

In the previous section, we covered how to set up your dataset and configure your training pipeline using a `Config` class. Now, let's dive into training your model and monitoring its progress using TensorBoard.

### Exploring Data Transformations and Augmentations

If you're looking for examples of data transformations and augmentations, you can explore the provided `notebook.ipynb` file. This notebook contains various examples of data preprocessing techniques, such as resizing, cropping, rotation, and more.

To open and explore the notebook:
1. Navigate to the directory containing the `notebook.ipynb` file.
3. Open the notebook and run the cells to see different transformation and augmentation examples.


### Training the Model

To train your model, you can use the provided `train.py` script. Make sure you have set up your environment correctly and installed all dependencies as mentioned earlier. Here's how you can run the training pipeline:

1. Open your terminal or command prompt.
2. Navigate to the directory containing the `train.py` script.
3. Run the following command:

```bash
python train.py
```

This command will execute the training script and start training your model based on the parameters specified in your `Config` class.

### Monitoring Training Progress with TensorBoard

TensorBoard is a powerful tool for visualizing and monitoring the training process. You can use it to track metrics such as loss, accuracy, and learning rate over time, as well as visualize model graphs and embeddings.

To load TensorBoard logs and monitor your training progress:

1. Ensure you have TensorBoard installed. You can install it via pip:

```bash
pip install tensorboard
```

2. Once your model starts training, TensorBoard logs will be generated in the specified directory (e.g., `"logs/"`). You can launch TensorBoard using the following command:

```bash
tensorboard --logdir=logs/
```

This command will start a TensorBoard server locally, allowing you to view your training metrics and visualizations in your web browser.


## Gradio - Diabetic Retinopathy Detection App
<!-- 
<iframe src="https://bhimrazy-diabetic-retinopathy-detection.hf.space" frameborder="0" width="1920" height="1080"></iframe>
-->
### Overview
Welcome to Diabetic Retinopathy Detection App! This app utilizes deep learning models to detect diabetic retinopathy in retinal images. Diabetic retinopathy is a common complication of diabetes and early detection is crucial for effective treatment.

### Try It Out
Use the interactive interface below to upload retinal images and get predictions on diabetic retinopathy severity.

[Open Diabetic Retinopathy Detection App](https://bhimrazy-diabetic-retinopathy-detection.hf.space)

[![Gradio App](https://github.com/bhimrazy/diabetic-retinopathy-detection/assets/46085301/4e0788dd-84a1-427e-a38a-e22c2aa86c50)](https://bhimrazy-diabetic-retinopathy-detection.hf.space)

### How to Use
1. Click on the "Open Diabetic Retinopathy Detection App" button above.
2. Upload a retinal image by clicking on the "Upload Image" button.
3. Once the image is uploaded, the model will process it and provide predictions on the severity of diabetic retinopathy.
4. Interpret the results provided by the model.

## License

[MIT](./LICENSE)


## Authors

- [@bhimrazy](https://www.github.com/bhimrazy)

