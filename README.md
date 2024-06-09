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
## Dataset

The dataset used in this project is based on the NEU surface defect dataset and additional images generated by Girish4474. The dataset consists of images of metal surfaces, which are categorized into two folders:

1. Clean: Images of metal surfaces without any anomalies, generated from cleaned metal surfaces.

Examples of clean surface

![clean - (23)](https://github.com/mhkt19/AnomalyDetection/assets/3819181/ad65a0d3-a71a-495a-984e-10d68bcf9efa)![clean - (9)](https://github.com/mhkt19/AnomalyDetection/assets/3819181/550d911a-766c-4424-ad3d-0f6b376c91fa)![clean - (72)](https://github.com/mhkt19/AnomalyDetection/assets/3819181/16781171-d2d6-4f05-95be-23c2c998e09e)



2. Faulty: Images of metal surfaces with scratches or other anomalies, including types such as "scratches", "pitted surface", "crazing", "inclusion", "rolled-in", and "patches".

Examples of faulty surface

![crazing_1](https://github.com/mhkt19/AnomalyDetection/assets/3819181/1afd5323-f102-42b3-9408-2e963c320e1d) ![inclusion_21](https://github.com/mhkt19/AnomalyDetection/assets/3819181/512b6af3-e2cf-4d67-bc46-395bca3dc442)![patches_26](https://github.com/mhkt19/AnomalyDetection/assets/3819181/bacc670e-212b-48ad-8a30-b877127347a3)

![pitted_surface (13)](https://github.com/mhkt19/AnomalyDetection/assets/3819181/d4a774e0-0a34-41c8-897f-57edadc655b7)![scratches_10](https://github.com/mhkt19/AnomalyDetection/assets/3819181/79e48e9f-431e-4f64-8f2e-de7d49319a36)![rolled-in_scale_32](https://github.com/mhkt19/AnomalyDetection/assets/3819181/ebee0841-1e45-43c3-83c8-1cc3889248e2)








## Result:
After training, the evaluation metrics such as accuracy, precision, recall, and confusion matrix are saved in metrics.txt. Since the train/test data is selected randomly in each run, the results are not deterministic across multiple runs. However, in most runs, the model achieves 100% accuracy on both the train and test data.

## Contributing:
Feel free to contribute to this project by opening issues or submitting pull requests.

## License:
This project is licensed under the MIT License.
