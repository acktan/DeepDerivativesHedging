# Natixis DeepHedging Project X-HEC DSB 2023

Recent progress achieved in data science and deep learning make a model independent approach for hedging possible
These hedging approaches well known as deep hedging are machine learning algorithms able to consider market frictions as well as trading constraints without using risk sensitivities metrics computed by pricing models.
The objective of the challenge is to replace classical hedging strategies founded on the calculation of risk sensitivities (Greeks) by machine learning algorithms.


## 🚀 Getting started with the repository

To ensure that all libraries are installed pip install the requirements file:

```
pip install - r requirements.txt
```

To run the model go to the console and run following command: 

```
python main.py
```

## 🗂 Repository structure

Our repository is structured in the following way

```
|natives_deephedging
   |--data
   |--output
   |-----evaluation
   |-----inference
   |-----model
   |--params
   |--src
   |-----evaluation
   |-----inference
   |-----loading
   |-----model
   |-----preprocessing
   |-----train
   |-----utils
   |--main.py
   |--README.md
   |--requirements.txt
```

### 📊 Data
The Data folder contains all dataset used to train the models
### ↗️ Output
In the Output folder we have the evaluation folder that includes the png of our explorative data analysis.
Inference folder contains the predicted deltas of our models. 
Model folder includes the two pest models that we used in the Black Scholes and Heston case.
### 🔢 Params 
Params folder includes the configuration file
### ℹ️ Src
#### Evaluation

The function *evaluate_model* is a method of the Evaluator class. The method takes in two inputs, train_loss and val_loss, which are lists of the training loss and validation loss respectively. The function plots the training loss and validation loss on the same graph, with the x-axis being the number of epochs and the y-axis being the loss. The function also saves the graph to a file in a specified directory.

The method *evaluate_train_dataset* takes in three inputs, model, S, v, payoff, and train_class. It first prepares the inputs, S and v by converting them to tensors and reshaping them. Then it calculates the loss by calling the loss method on train_class and passing in the model's predictions, deltas, the inputs, S, payoff, var, and costs as arguments. It also calculates a risk measure by calling the evaluation method on train_class and passing in the loss as an argument. The function then prints the risk measure calculated on the full training set.

#### Inference

The Inference class is used to make predictions on a test set, load test data, and save the predictions. The class takes in two inputs, conf which is a configuration file and model which is the trained model.

The *load_test_data_bs* method is used to load test data for the Black-Scholes model. It reads in a CSV file containing the test data and converts it into a numpy array. The array is then converted into a torch tensor and returned.

The *load_test_data_hest* method is used to load test data for the Heston model. It reads in two CSV files, one for the stock prices and one for the variance swap, converts them into numpy arrays, and then converts them into torch tensors. It then returns the two tensors.

The *predict* method takes in no inputs, and if the model used is the Black-Scholes model, it loads the test data, performs predictions on the test data using the trained model and returns the predictions as a Pandas dataframe. If the model is Heston model, it loads the test data, performs predictions on the test data using the trained model, separates the predictions into two dataframes, one for stock prices and one for variance swap and returns the two dataframes.

The *save_predictions* method takes in no inputs and is used to save the predictions in a CSV file. It creates a new CSV file in a specified directory, and saves the predictions in the file.

#### Loading






## 📫 Contacts LinkedIn 

If you have any feedback, please reach out to us on LinkedIN!!!

- [Lea Chader](https://www.linkedin.com/in/lea-chader/)
- [Inès Benito](https://www.linkedin.com/in/ines-benito/)
- [Kun Tan](https://www.linkedin.com/in/kun-tan/)
- [Milos Basic](https://www.linkedin.com/in/milos-basic/)
- [Salah Mahmoudi](https://www.linkedin.com/in/salahmahmoudi/)
- [Michael Liersch](https://www.linkedin.com/in/michael-liersch/)

