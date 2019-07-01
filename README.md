# Char RNN PyTorch

Minimalist code for character-level language modelling using LSTM in PyTorch. The RNN is trained to predict next letter in a given text sequence. The trained model can then be used to generate a new text sequence resembling the original data.

## Requirements

Trained and tested on:

- `Python 3.6`
- `PyTorch 1.0`
- `NumPy 1.16.3`

## Usage

### Training
To train a new network run `CharRNN.py`. If you are using custom data, make sure you change the `data_path` and `save_path` variables accordingly. To keep the code simple the batch size is one, so the training procedure is a bit slow. The average loss and a sample from the model is printed after every epoch.

### Testing
To test a preTrained network (trained on Shakespeare.txt in the data folder) run `test.py`. The training dataset is required for testing too for creating vocabulary dictionary, and also for sampling a random small (10 letters) text sequence to begin generation.


## Acknowledgements
This code is based on the [char-rnn](https://github.com/karpathy/char-rnn) and [min-char-rnn](https://gist.github.com/karpathy/d4dee566867f8291f086) code by Andrej Karpathy, which is in turn based on Oxford University Machine Learning class [practical 6](https://github.com/oxford-cs-ml-2015/practical6), which is in turn based on [learning to execute](https://github.com/wojciechz/learning_to_execute) code from Wojciech Zaremba.

