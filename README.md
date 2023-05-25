
# Dog_vs_Cat Classification -> Support Vector Machine

Thise project motive to Explore image processing model using the Dog_Vs_Cat classification datasets mobilenet_model.





## ⌛Installation

Kaggle environment Installation

```bash
  ! pip install kaggle

  from google.colab import files
  !ls -lha kaggle.json
  !pip install -q kaggle
  !mkdir -p ~/.kaggle
  !cp kaggle.json ~/.kaggle/
  !pwd

  !chmod 600 ~/.kaggle/kaggel.json
  !kaggle datasets list

  !kaggle competitions download -c dogs-vs-cats
```
    
## API Reference

#### Get all items
## [Numpy](https://numpy.org/doc/stable/reference/)
## [Pandas](https://pandas.pydata.org/docs/reference/index.html)
## [Sciket-learn](https://scikit-learn.org/stable/modules/classes.html)
## [Mobile-Net](https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4)


# Steps fro model creation

1.Gather and clean the data: Collect the data that you will use for training and testing your model. This data may be in various formats such as CSV, Excel, or a database. You will need to clean the data by removing missing values, removing outliers, and transforming the data as necessary.

2.Split the data into training and testing sets: Split your data into two sets: one for training the model and another for testing the model. The typical split is 80% for training and 20% for testing, but this can vary depending on the size of the dataset.

3.Feature selection: Select the features or independent variables that will be used to train the model. It's important to choose features that are relevant to the problem you are trying to solve.

4.Scaling and normalization: Scale and normalize the data to ensure that each feature has a similar range and distribution. This step is important because linear regression is sensitive to the scale of the features.

5.Train the model: Use the training dataset to train the linear regression model. During training, the model will learn the coefficients for each feature that will be used to make predictions.

6.Evaluate the model: Use the testing dataset to evaluate the performance of the model. You can use metrics such as mean squared error, R-squared, and others to evaluate the performance of the model.

7.Tune the model: If the model's performance is not satisfactory, you can tune the hyperparameters such as the learning rate or regularization parameter to improve the model's performance.

8.Deploy the model: Once the model has been trained and tested, you can deploy it to make predictions on new data.

## 📝Description

* Thise projects based on LinearRegression algorithm.
* In thise projects i have using Support vector machine model
* 1.SVC() Using sklearn machine learning libraries.


## 📊Datasets
## [Dounload Datasets](https://drive.google.com/drive/folders/19R7Bo7LPMNfxO3FCDtmAbhgO7BAaQhnp)
* Download the datasets for costom training


## 🎯Inference Demo

```bash
  import tensorflow as tf
  import tensorflow_hub  as hub

  mobilenet_model = 'https://tfhub.dev/google/tf2-preview/   inception_v3/feature_vector/4'
```

## 🕮 Please go through [Dog_vs_Cat_classification.docx]() more info.

