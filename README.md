# Project Lots of Data

<p>Predicting if a game of chess on Chess.com will result in a win or loss a game.

To devise a proper way to rectify the current model, which considers that the game probability to be only dependant on Difference in ELO, following pre-game parameters were used:
1. Difference in *ELO Rating*
2. Assigned Color
3. Time allowed per player (aka game time)
4. Time of the Day
5. Day of the Week

To properly assess which of these factors played the biggest impact, the decision to try different combinations was adopted. Here are all the combinations possible, represented as thier number on the list above:
- Single parameters only (1, 2, 3, 4, 5)
- Combination of two parameters (1+2, 1+3, 1+4, 1+5, 2+3, 2+4, 2+5, 3+4, 3+5, 4+5)
- Combination of three parameters (1+2+3, 1+2+4, 1+2+5, 1+3+4, 1+3+5, 1+4+5, 2+3+4, 2+3+5, 2+4+5, 3+4+5)
- Combination of four parameters (1+2+3+4, 1+2+3+5, 1+2+4+5, 1+3+4+5, 2+3+4+5)
- Combination of all five parameters (1+2+3+4+5)

This ended up to be 31 different combinations. Ideally all of them should be ran to predict which one of these will end up being the best estimators. However, after running only a handful of tests, all the parameters devoid of Difference in ELO were dropped, since the accuracy in none of these combinations was even close to 66%.

That left 16 total parameters to run tests on:
- Single parameters only (1)
- Combination of two parameters (1+2, 1+3, 1+4, 1+5)
- Combination of three parameters (1+2+3, 1+2+4, 1+2+5, 1+3+4, 1+3+5, 1+4+5)
- Combination of four parameters (1+2+3+4, 1+2+3+5, 1+2+4+5, 1+3+4+5)
- Combination of all five parameters (1+2+3+4+5)
</p>

## Data:
The initial data contained a total of 2,276 total games played. All the games that ended up with the result of a draw were dropped, since the main idea is to predict a win or a loss.

Hence, 2,127 games were taken as the dataset for training and testing. 

With a small subset of games data, train test split was made based on personal experience. The test set was 100 games and the remainder were used for training (2027 games). This is a roughly 95/5 split.

To make the prediction model perform better, difference in ELO and the time of the day were binned.

The time of the day was easy to bin, which was to bin it in to each hour of the day (Total of 24 bins.)

To properly bin the difference in ELO, a different aproach was used. The bins were created in the following manner:
- Intervals of 100 for difference in ELO being above 600 or below -600
- Intervals of 50 for difference in ELO being between 400 to 600 and  between -600 and -300
- Intervals of 25 for difference in ELO being between 250 to 325 and between -250 and -200
- Intervals of 10 for difference in ELO being between 110 to 200 and between -200 and -100
- Intervals of 5 for difference in ELO being between -100 to 100
    
All bins labels were calculated and assigned roughly based on winning probability using original chances of winning ELO equation:
```
1/(1+10^m)<br>
m = (elo difference)/400
```

Also, dummy columns were created for:
- Time of the day
- Game time
- Day of the week

## Model:

The Keras Sequential Model was used to determine which combination of these parameters gave the best results.

The Keras Classifier created has the following layers:
- Input layer:
    - Activation Function = SoftMax
    - Units = 64
- First hidden Layer:
    - Activation Function = ReLu
    - Units = 128
- Second (and final) hidden layer:
    - Activation Function = SoftMax
    - Units = 32
- Output layer:
    - Activation Function = Sigmoid

To compile the classifier, 'accuracy' (aka 'binary_accuracy') was determined for the to be the best option for the metrics to determine the validity of the model, since it has to determine only between win or lose.

Idea of Grid Search was rejected, primarilly because of the amount of data available. Since the training set is of only 2,027 games, the grid-search will be using 10% to 20% of it as validation data, and being a fairly beginner at chess, the playing style is ever so evolving, any part of the data ommitted will lead to picking the wrong parameters.

Even if the validation data of the 100 test games is given, the results would not be displayed in detail as you would see in the raw results document.

Hence, to get a better idea of which parameters will work the best, "For-Loop" is used to train the classifier with different parameters. These following parameters are used in that for loop:

1. Losses:
    1. Mean Absolute Error (Measure of difference between two continuous variables)
    2. Binary Crossentropy (AKA Minimizing Log Loss)
    3. Meas Square Error (Mean squared difference between the estimated values and what is estimated)
2. Optimizers:
    1. Nadam (Adam RMSprop with Nesterov momentum)
    2. RMSProp (Divides the gradient by running mean of recent magnitude)
    3. AdaGrad (optimizer with parameter-specific LR, adapted relative to how frequently a parameter gets updated during training)
    4. Adam (ADAptive Moment estimation)
    5. AdaDelta (Adapts LR based on a moving window of gradient updates)
3. Batch Sizes:
    1. 8
    2. 20
    3. 44
    4. 92
4. Epochs:
    1. 50
    2. 100
    3. 200
    
Please note, that SGD, Squared Hinge, and Hinge were also tried, but they all failed to give any resonable predictions. All of them predicted all wins or all losses, hence were dropped after running handful of tests.

Currently all tests are being run on different machines (Kaggle Kernel, personal computer, and AWS EC2.)

Some other/simpler models are also being used to determine the best way to predict the outcome. Some other models include: 

1. Linear Discriminant Analysis
2. Quadratic Discriminant Analysis
3. Gaussian Process Classifier
4. Logistic Regression
5. Ada Boost Classifier
6. Stochastic Gradient Descent Classifier
7. Ridge Classifier
8. K Neighbors Classifier
9. Multi-layer Perceptron classifier

The results will be published in the "docs" folder once the ideal combinations and model have been determined.

## Usage
Clone this repository with the command

```
git clone git@github.com:Data-is-life/lots-of-data.git
```

All codes are located in **src** folder.

## Repository Structure
The repository has the following structure.

```
├── data
│   ├── dest.pgn
│   ├── main_with_all_info.csv
│   ├── moves.csv
│   ├── moves_initial.csv
│   ├── use_for_analysis.csv
│   └── use_for_predictions.csv
├── docs_&_notes
│   ├── calculations for bins.ods
│   ├── Dropped_models.md
│   ├── first_ann_run_results.csv
│   ├── first_run_raw_adagrad_mae.md
│   └── first_run_raw_nadam_mae.md
├── README.md
└── src
    ├── all_classification_models.ipynb
    ├── all_graphs.py
    ├── chess_analysis.ipynb
    ├── chess_ann.ipynb
    ├── clean_chess_game_log.py
    ├── complete_keras_model.py
    ├── dummies_bins_test_train_cv.py
    ├── old_files
    │   ├── Chess_KNN_Crossval.ipynb
    │   ├── chess_linear.ipynb
    │   ├── chess_logistic.ipynb
    │   ├── Chess + other .txt
    │   ├── Old_chess_data_cleanup.ipynb
    │   └── Unused Chess comands.txt
```
## Tools used so far:

- Pandas
- Numpy
- Scikit Learn
- Keras
- Tensorflow
- PyTorch

<img src="https://www.python.org/static/community_logos/python-logo-master-v3-TM.png" width="300"></br>
<img src="https://pandas.pydata.org/_static/pandas_logo.png" width="300"></br>
<img src="https://bids.berkeley.edu/sites/default/files/styles/400x225/public/projects/numpy_project_page.jpg?itok=flrdydei" width="300"></br>
<img src="https://www.scipy-lectures.org/_images/scikit-learn-logo.png" width="300"></br>
<img src="https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png" width="300"></br>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/2000px-TensorFlowLogo.svg.png" width="300"></br>
<img src="https://pytorch.org/docs/stable/_static/pytorch-logo-dark.svg" width="300">


```
Copyright (c) 2018, Mohit Gangwani

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

**Stay tuned, there is more to come!**
