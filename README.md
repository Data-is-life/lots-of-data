# Project Lots of Data

<p> Predicting if I will win or lose a game on Chess.com based on:

- Difference in *ELO Rating*
- Length of the game
- Assigned Color
- Time of the day
- Day of the week

The purpose of this project is to  __improve my chess game__.</p>

## Data:
All the data is from Chess.com where I play.

## Model Selection:

Models you'll find that I'm working on:

1. Artificial Neural Network
2. Recurrent Neural Network
3. Linear Discriminant Analysis
4. Quadratic Discriminant Analysis
5. Gaussian Process Classifier
6. Logistic Regression
7. Ada Boost Classifier
8. Stochastic Gradient Descent Classifier
9. Ridge Classifier
10. K Neighbors Classifier
11. Multi-layer Perceptron classifier

Best accuracies for test set so far are around 80%. Goal is to have accuracy roughly 90%. 

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
│   ├── adams_kasparov_2005.pgn
│   ├── dest.pgn
│   ├── df_final.csv
│   ├── df_for_analysis.csv
│   ├── df_for_model.csv
│   ├── moves.csv
│   └── moves_initial.csv
├── docs_&_notes
│   ├── Chess + other .txt
│   └── Unused Chess comands.txt
├── Dropped_models.md
├── README.md
└── src
    ├── All_in_one.ipynb
    ├── Chess_analysis.ipynb
    ├── Chess_ann.ipynb
    ├── Chess_clean_data.ipynb
    ├── Chess_KNN_Crossval.ipynb
    ├── chess_linear.ipynb
    ├── chess_logistic.ipynb
    ├── Chess_rnn.ipynb
    ├── df_functions.py
    ├── Old_chess_data_cleanup.ipynb
    └── Score.ipynb
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
