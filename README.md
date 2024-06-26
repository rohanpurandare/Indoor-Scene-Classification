# Exploiting Local and Global Discriminative Information for Indoor Scene Recognition

## ABSTRACT
Classifying indoor scenery remains a significant challenge in computer vision. While Convolutional Neural Networks (CNNs) have significantly transformed conventional methodologies in this regard, our research aims to investigate the resilience of traditional techniques, with a specific focus on refining a distinctive dimension: object definition within the scenery. By incorporating object definition as a key feature in the training process, we intend to conduct a comprehensive benchmarking study, comparing the efficacy of traditional methods against the advancements introduced by newer techniques in this domain.
## 1 PROBLEM DEFINITION
The primary goal of this project is to conduct an exhaustive benchmarking study to examine the impact of various classification models and sampling strategies on key evaluation metrics such as accuracy, precision, recall, and F1 score. Several recent approaches have attempted to improve indoor scene classification through the integration of relative spatial relationships and object identification. Despite the advancements, achieving accurate scene recognition remains a formidable challenge. Therefore, the main question we want to answer is how effective traditional methodologies are in comparison to these newer techniques, specifically focusing on whether traditional approaches can perform as well as or better than modern approaches. Through this investigation, we hope to learn more about the advantages and disadvantages of various classification techniques, which will further the field of indoor scene classification research and possibly direct future advancements in this field.
## 2 PLAN OF ACTIVITIES
### 2.1 Sampling Methods
We will employ three widely recognized sampling techniques: Simple Random Sampling, Stratified Sampling, and Cluster Sampling. By combining these techniques with varying batch sizes, we aim to gain insights into how the model’s training and performance are affected by different data selection approaches.
### 2.2 Training Approach 
In the training phase, we plan to utilize pre-trained Object Semantic Segmentation models, specifically YOLOv81. Subsequently, the outputs from the Segmentation Model will be fed into three well-known classification models: K-Nearest Neighbours (KNN), Support Vector Machines (SVMs), and Convolutional Neural Networks (CNNs).
### 2.3 Testing Methodology
We will set aside an independent subset of the datasets being used and will assess the models on this. At the project’s conclusion, we expect to have nine distinct models trained based on different sampling techniques. The results will be evaluated and interpreted, and comparative analyses against state-of-the-art models will be conducted to gauge the efficacy of traditional models in this context.
## 3 EVALUATION
Our goal is to perform a bench-marking study to evaluate the performance of traditional models against existing models in the field. As such, we will analyze the accuracy of each of our models, as well as their properties, advantages, and disadvantages, and compare them against that of state-of-the-art models in the problem space.
### 3.1 Datasets
We will be using the MIT Indoor Dataset2, Places3653 and SUN3974 datasets for training, validation and testing. We will also be concatenating all three datasets to get an extensive view during training and testing.
### 3.2 Metrics Selection
Our comprehensive evaluation of the models after implementation will involve a thorough analysis with a set of well-established metrics. The chosen metrics include the confusion matrix, accuracy, precision, recall, and F1 score
### Rationale for Metrics
The confusion matrix will provide a detailed breakdown of the model’s predictions, allowing us to discern true positives, true negatives, false positives, and false negatives. Accuracy will offer a general measure of the model’s correctness, while precision will focus on the accuracy of positive predictions. Recall, on the other hand, will emphasize the model’s ability to correctly identify all relevant instances. The F1 score, as a harmonic mean of precision and recall, will provide a balanced assessment of the model’s performance.

_Please note: This was done in a group project with Shruti Goyal, Arnav Arora, and Harmya Bhatt_
