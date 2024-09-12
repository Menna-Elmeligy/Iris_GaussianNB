# Classification of Irises Using Gaussian Naive Bayes Classifier  

## Project Overview
This project provides an analysis and classification of the Iris dataset using *Scikit-learn*. The [**Iris Dataset**](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) is one of the most famous datasets in machine learning and statistics and is often used as a beginner's dataset for classification problems. 

The aim of this project is to classify irises into either _"Setosa", "Versicolour"_ or _"Virginica"_, based on their petal width, petal length, sepal length and sepal width using **GaussianNB classifier**.

## How to use?

1. Clone the repository to your local machine:  
`git clone https://github.com/Menna-Elmeligy/Iris_GaussianNB.git`  

2. Navigate to the project directory:  
`cd Iris_GaussianNB`  

3. Create a virtual environment using conda:  
```conda create --name iris_classifier python
conda activate iris_classifier
```

4. Install the required dependencies.  
`pip install -r requirements.txt`

5. The notebook is structured to guide you through the steps of:

* Loading the Iris dataset.
* Performing basic exploratory data analysis (EDA).
* Preprocessing the data.
* Training a machine learning model to classify the different Iris species.
* Evaluating the model's performance.

5. If you have a given set of measurements in cm for an iris and want to classify it, follow the following steps:
```# Load the model from the file
model = joblib.load('iris_NB_classifier.pkl')

# Define new petal and sepal measurements
x1, x2, x3, x4 = sepal_length, sepal_width, petal_length, petal_width
new_sample = [[x1, x2, x3, x4]]

# Make a prediction using the loaded model
predicted_class = model.predict(new_sample)
print(f"The predicted class is {predicted_class}")

# Map the predicted class to the actual species name
predicted_species = iris.target_names[predicted_class][0]
print(f"The predicted species is: {predicted_species}")
```
## Contributing
If you would like to contribute to this project, feel free to submit a pull request or open an issue.

