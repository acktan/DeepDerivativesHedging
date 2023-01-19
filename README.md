 # Natixis DeepHedging Project X-HEC DSB 2023

Recent progress achieved in data science and deep learning make a model independent approach for hedging possible
These hedging approaches well known as deep hedging are machine learning algorithms able to consider market frictions as well as trading constraints without using risk sensitivities metrics computed by pricing models.
The objective of the challenge is to replace classical hedging strategies founded on the calculation of risk sensitivities (Greeks) by machine learning algorithms.


## 🚀 Getting started with the repository

To ensure that all libraries are installed pip install the requirements file:

```
pip install -r requirements.txt
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
In the Output folder we have the evaluation folder that includes the validation and training loss figure as evaluation of our train model.
Inference folder contains the predicted deltas of our models. 
Model folder includes the two best models that we used in the Black Scholes and Heston case.
### 🔢 Params 
Params folder includes the configuration file and a logs.log file that is added to view the log info and debug
### ℹ️ Src

The Src folder contains all the different classes combined in the main.py file. The following is a description of all classes used.

#### Evaluation

The function *evaluate_model* is a method of the Evaluator class. The method takes in two inputs, train_loss and val_loss, which are lists of the training loss and validation loss respectively. The function plots the training loss and validation loss on the same graph, with the x-axis being the number of epochs and the y-axis being the loss. The function also saves the graph to a file in a specified directory.

The method *evaluate_train_dataset* takes in three inputs, *model*, *S*, *v*, *payoff*, and *train_class*. It first prepares the inputs, *S* and *v* by converting them to tensors and reshaping them. Then it calculates the loss by calling the loss method on *train_class* and passing in the model's predictions, deltas, the inputs, *S*, *payoff*, *var*, and *costs* as arguments. It also calculates a risk measure by calling the evaluation method on *train_class* and passing in the loss as an argument. The function then prints the risk measure calculated on the full training set.

#### Inference

The *Inference* class is used to make predictions on a test set, load test data, and save the predictions. The class takes in two inputs, conf which is a configuration file and model which is the trained model.

The *load_test_data_bs* method is used to load test data for the Black-Scholes model. It reads in a CSV file containing the test data and converts it into a numpy array. The array is then converted into a torch tensor and returned.

The *load_test_data_hest* method is used to load test data for the Heston model. It reads in two CSV files, one for the stock prices and one for the variance swap, converts them into numpy arrays, and then converts them into torch tensors. It then returns the two tensors.

The *predict* method takes in no inputs, and if the model used is the Black-Scholes model, it loads the test data, performs predictions on the test data using the trained model and returns the predictions as a Pandas dataframe. If the model is Heston model, it loads the test data, performs predictions on the test data using the trained model, separates the predictions into two dataframes, one for stock prices and one for variance swap and returns the two dataframes.

The *save_predictions* method takes in no inputs and is used to save the predictions in a CSV file. It creates a new CSV file in a specified directory, and saves the predictions in the file.

#### Loading
The *DataLoader* class is used to load the data for training a model. It takes in one input, conf which is a configuration file. The class contains several methods that are used to process and load the training data.

The *col_numeric_names* method takes in a dataframe as an input, and returns the same dataframe with column names from 0 to 30.

The *absolute_growth* method takes in a dataframe containing stock prices as an input and returns a dataframe containing the absolute growth in stock price for each path in the dataframe.

The *percentage_growth* method takes in a dataframe containing stock prices as an input and returns a dataframe containing the percentage growth in stock price for each path in the dataframe.

The *return_filenames* method takes in no input and returns the names of files within the input directory.

The *get_train_data* method is used to return the training data for Black & Scholes or Heston model. It reads in a CSV file containing the stock prices, variance swap (only for Heston) and payoffs. It returns the training data in the form of dataframes containing stock prices, variance swap and payoffs.

#### Model

The *DeepHedging_BS* and *DeepHedging_Hest* classes are used to create the model architecture for the Black-Scholes (BS) and Heston models respectively. Both classes are subclasses of the torch.nn.Module and they use the Pytorch library to define the architecture.

The *DeepHedging_BS* class initializes the following:

An RNN (Recurrent Neural Network) layer with input size, hidden size and number of layers as specified in the config file.
A linear layer with hidden size and output size as specified in the config file.
The DeepHedging_Hest class initializes the following:

   - An RNN (Recurrent Neural Network) layer with input size, hidden size and number of layers as specified in the config file.
   - A linear layer with hidden size and output size as specified in the config file.

Both the classes have a forward function that takes in a tensor S and processes it through the initialized layers. The output of the linear layer is returned and the size of the output is reduced by one dimension.

#### Preprocessing

The *DataPreprocessor* class is used to preprocess and split the training data into validation and training sets, and return dataloaders for both sets. The class takes in four inputs: conf which is a configuration file, df_train which is a dataframe with stock prices for the train set, pay_off which is a dataframe with payoffs for the train set, and df_growth which is a dataframe with absolute growth of stock prices for the train set.

The *train_val_split* method takes in one input, split_percent, which is the percentage of data that is for training, and returns four or eight outputs depending on the model used, train_X and val_X, which are training and validation tensors of stock prices, train_payoff and val_payoff, which are training and validation tensors of payoffs, train_costs and val_costs, which are training and validation tensors of costs, and if the model used is Heston, train_var and val_var, which are training and validation variance swap.

The *get_train_val_dataloader* method takes no inputs and returns two outputs, train_loader and val_loader, which are torch dataloaders for the training and validation sets. The dataloaders are created by creating a TensorDataset for the train and val sets, and then creating a DataLoader for each set with a specified batch size.

#### Train

The *Train* class is used to train an RNN model, either the Black-Scholes (BS) model or the Heston model. It takes in the config file, the model, the training dataloader and the validation dataloader as inputs. The class has several methods, including "loss()" that calculates the loss incurred by the predicted deltas, and "risk_measure()" which calculates a custom loss function for the BS Model. The "Train()" method is responsible for training the model, it uses the Adam optimizer, and it prints out the training and validation losses for each epoch. The class also has a method called "test()" which can be used to evaluate the performance of the trained model on unseen data. The model is trained on a device specified as CUDA if available or CPU.

### ❤️ Main.py

This python file serves as the main script and heart of the repository that runs the entire pipeline for the project. It starts by loading the config file and setting the random seed. Then it calls the DataLoader class to load the stock prices, variances and payoff. Next, it calls the DataPreprocessor class to split the data into training and validation sets and create dataloaders. Then, it calls the model class to create the model architecture. Then, it calls the Train class to train the model and save it or load a saved model. Next, it calls the Inference class to make predictions and saves them. Finally, it calls the Evaluator class to evaluate the performance of the model. The script also has a try-except block to catch any errors that might occur during the execution and log them. The script also has various debug statements to log the time taken for each step of the pipeline.


## 📫 Contacts LinkedIn 

If you have any feedback, please reach out to us on LinkedIN!!!

- [Lea Chader](https://www.linkedin.com/in/lea-chader/)
- [Inès Benito](https://www.linkedin.com/in/ines-benito/)
- [Kun Tan](https://www.linkedin.com/in/kun-tan/)
- [Milos Basic](https://www.linkedin.com/in/milos-basic/)
- [Salah Mahmoudi](https://www.linkedin.com/in/salahmahmoudi/)
- [Michael Liersch](https://www.linkedin.com/in/michael-liersch/)

