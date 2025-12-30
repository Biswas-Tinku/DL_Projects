# Image_recognization
Vision AI fundamentals: Building a image recognizer from scratch

# Fashion-MNIST Classification Project

This project aims classifying images from the Fashion-MNIST dataset using deep learning models. I am explore the impact of increasing model complexity, from a simple Artificial Neural Network (ANN) to more sophisticated Convolutional Neural Networks (CNNs), on performance and efficiency.

**About the Data**

The MINST dataset stands for `"Modified National Institute of Standards and Technology"`.  
The Fashion-MNIST (F-MNIST) dataset is a widely-used collection of Zalando's article images intended as a modern, more challenging replacement for the original handwritten digits MNIST dataset. It is a standard benchmark for evaluating machine learning algorithms, particularly in image classification tasks.   
**Key Characteristics of the dataset**
- `Content:` 70,000 grayscale images of fashion and clothing items.
- `Image Format:` Each image is low-resolution, sized at 28x28 pixels, with pixel values ranging from 0 to 255.
- `Dataset Splits:` It is divided into a training set of 60,000 images and a test set of 10,000 images.
- `Classes:` The images are categorized into 10 distinct classes.
- `Purpose:` F-MNIST is often the "Hello, World" for machine learning programs in computer vision, used to test and debug code, and to benchmark the performance of models like Convolutional Neural Networks (CNNs).   

**The 10 Classes**  

Each image is assigned a label from 0 to 9, corresponding to a specific type of clothing item: 

0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

**Project Agenda & Steps:**

I am using Tenserflow to create the neural networks for this project

1.  **Dataset Setup:**
    *   Import necessary libraries.
    *   Load the Fashion-MNIST dataset.
    *   Preprocess the data (normalize and reshape images, one-hot encode labels).
    *   Verify the shapes of the processed data.

2.  **Model Building:**
    *   Define the architecture for each model:
        *   Basic ANN Model
        *   Basic CNN Model
        *   Deeper CNN Model

3.  **Model Training:**
    *   Train each model using the prepared training data.
    *   Implement Early Stopping and Model Checkpointing to optimize training and save the best model weights.

4.  **Model Evaluation:**
    *   Load the best weights for each trained model.
    *   Evaluate each model's performance on the test set using metrics like loss and accuracy.
    *   Visualize the training history (accuracy and loss curves) for comparison.
    *   Generate and visualize confusion matrices to understand model performance on each class.

5.  **Prediction Analysis:**
    *   Use the best performing model (Basic CNN in this case) to make predictions on the test set.
    *   Identify and visualize examples of both correctly and incorrectly classified images to gain insights into model behavior.

**Goal:** To analyze and demonstrate how model complexity influences classification accuracy and efficiency on the Fashion-MNIST dataset.

**Summary of Work:**

1.  **Data Preparation:** The Fashion-MNIST dataset was loaded, normalized, reshaped, and one-hot encoded, preparing it for use with the different model architectures.
2.  **Model Development:** Three models of increasing complexity were defined: a basic Artificial Neural Network (ANN), a basic Convolutional Neural Network (CNN), and a deeper CNN with additional layers, batch normalization, and dropout.
3.  **Model Training:** Each model was trained using the preprocessed training data with Early Stopping and Model Checkpointing to prevent overfitting and save the best performing weights based on validation loss.
4.  **Model Evaluation:** The trained models were evaluated on the test set. Performance metrics (loss and accuracy) were calculated, and the training history was visualized. Confusion matrices were generated to analyze class-specific performance.
5.  **Prediction Analysis:** Predictions were made using the Basic CNN model, and examples of correctly and incorrectly classified images were visualized to gain insights into the model's strengths and weaknesses.

**Key Findings and Conclusion:**

Based on the evaluation results:

*   The **Basic CNN model** generally achieved the best balance of performance (highest accuracy, lowest loss) on the test set compared to the ANN and Deeper CNN models.
*   The **ANN model** performed reasonably well but was outperformed by both CNN architectures, highlighting the advantage of convolutional layers for image classification tasks.
*   The **Deeper CNN model**, despite its increased complexity, did not consistently outperform the Basic CNN model on this dataset. This could be due to various factors such as the dataset size, the architecture choices, or the regularization applied. For this particular task and dataset, the increased complexity of the deeper model might not have been necessary or could have led to some overfitting despite the regularization techniques.

In conclusion, the Basic CNN model demonstrated superior performance for this Fashion-MNIST classification task, suggesting that a moderate level of complexity with convolutional layers is effective for this dataset. Further tuning of hyperparameters or architectural variations might potentially improve performance across all models, but the current results clearly show the benefits of CNNs over ANNs for this image data.

**Snapshot Of the dataset**
![alt text](Dataset_raw-1.jpg)
**Snapshot of the prediction**
![alt text](Prediction-1.jpg)
