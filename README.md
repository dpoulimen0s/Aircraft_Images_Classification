# Aircraft Classification with ResNet50

This repository provides a solution for fine-grained classification of aircraft images using the ResNet50 deep learning model. The workflow involves preprocessing image data, training a neural network, hyperparameter tuning, and evaluating performance metrics. Key highlights are detailed below:

## **Features**
- **Dataset Handling**:  
  - Custom `AircraftDataset` class for loading and preprocessing images and labels from CSV files.
  - Applied transformations: resizing, cropping, normalization.

- **Model Architecture**:  
  - ResNet50 pretrained on ImageNet for feature extraction.  
  - Final fully connected layer adapted to classify 100 aircraft categories.  

- **Training and Validation**:  
  - Cross-entropy loss for optimization.  
  - Adam optimizer with learning rate scheduling for fine-tuning.  
  - Early stopping mechanism to prevent overfitting.

- **Hyperparameter Tuning**:  
  - Explored combinations of learning rates and batch sizes.  
  - Best-performing model saved based on validation loss.

- **Evaluation Metrics**:  
  - Accuracy, precision, recall, F1 score, and confusion matrix calculated on the test set.  
  - Model achieved approximately **75.73% accuracy**.

## **Setup**
1. Clone the repository and set up the dataset in Google Drive.  
2. Modify dataset paths in the notebook to reflect your directory structure.  
3. Run the notebook in Google Colab to train and evaluate the model.  

## **Results**
- **Metrics**:  
  - Accuracy: ~75.73%  
  - Precision: ~79.36%  
  - Recall: ~75.72%  
  - F1 Score: ~75.75%  

- **Confusion Matrix**: Highlights performance across 100 aircraft categories.

## **Requirements**
- Python 3.7+
- PyTorch
- Torchvision
- NumPy, Pandas, Scikit-learn
- Google Colab for GPU support

## **How to Use**
1. Train the model by running the training notebook.  
2. Evaluate the best model on the test set.  
3. Use the demo function for further experimentation or classification tasks. 

For more details, please refer to the code and notebook files in this repository.
