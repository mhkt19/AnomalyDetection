# Anomaly Detection

This project aims to develop a neural network-based classifier to detect anomalies (scratches) on metal surfaces using PyTorch. The dataset consists of images of metal surfaces categorized into two classes: `clean` and `faulty`.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- PIL (Python Imaging Library)
  
## Evaluate the Model:
The script will automatically and randomly split the dataset into training (80%) and test sets (20%) .
It will train the model and save the trained model to model.pth.
Misclassified images will be saved in the test_images/misclassified/FP and test_images/misclassified/FN directories.
Evaluation metrics (accuracy, precision, recall, confusion matrix) will be saved in metrics.txt.
Model
The model used for this project is a pre-trained ResNet-18, fine-tuned for binary classification (clean vs faulty).

## Result:
After training, the evaluation metrics such as accuracy, precision, recall, and confusion matrix are saved in metrics.txt.

## Contributing:
Feel free to contribute to this project by opening issues or submitting pull requests.

## License:
This project is licensed under the MIT License.
