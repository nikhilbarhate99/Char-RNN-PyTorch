# Char RNN PyTorch

Minimalist code for character-level language modelling using Multi-layer Recurrent Neural Networks (LSTM) in PyTorch. The RNN is trained to predict next letter in a given text sequence. The trained model can then be used to generate a new text sequence resembling the original data.

## Requirements

Trained and tested on:

- `Python 3.6`
- `PyTorch 1.0`
- `NumPy 1.16.3`

## Usage

### Training
To train a new network run `CharRNN.py`. If you are using custom data, change the `data_path` and `save_path` variables accordingly. To keep the code simple the batch size is one, so the training procedure is a bit slow. The average loss and a sample from the model is printed after every epoch.

### Testing
To test a preTrained network (~15 epochs) run `test.py`. The training dataset is required for testing, to create vocabulary dictionary, and also for sampling a random small (10 letters) text sequence to begin generation.

## Samples

**Shakespeare Dataset (~ 15 epochs) :**
```
DOCSER:
What, will thy fond law?
or that in all the chains that livinar?

KING HENRY V:
Come, come, I should our name answer'd for two mans
To deafly upbrain, and broke him so our
Master Athital. Mark ye, I say!

B-CANSSIO:
Come, let us die.

Hostes:
This was my prince of holy empress,
That shalt thou save you in it with brave cap of heaven.
Or is the digest and praud with their closets save of faitral'?

KING HENRY V:
Your treason follow Ncpius, Dout &ystermans' clent,
On the pity can, when tell them
Freely from direen prisoners town; and let us
know the man of all.

FLUELLEN:
Go tell you.
```

-----------------------------------------------------------------

**Sherlock Holmes Dataset (~ 15 epochs) :**
```
 Mr. Holmes had drawn up and again so brick, at west who closed upon
 the loud broken pallow and a cabmon ta the chair that we had fired
 out.

 "I wished in," said Holmes sobbily, "trust in the light. I said that you
 have to do with Gardens, come, you will pass you
 light, so you print?"

 "We are it is impossible."

 "I know that so submer a case here did he give you after I
 tell you?"

 "Ah, sir, I keep them, Watson," I said a tueler
 inspectoruded upon either way. "Home!" said Admirable
 Street. "But not considered a memory, which it was to complice him."

 I had so vallemed found me about this gloomy men.
```


## Acknowledgements
This code is based on the [char-rnn](https://github.com/karpathy/char-rnn) and [min-char-rnn](https://gist.github.com/karpathy/d4dee566867f8291f086) code by Andrej Karpathy, which is in turn based on Oxford University Machine Learning class [practical 6](https://github.com/oxford-cs-ml-2015/practical6), which is in turn based on [learning to execute](https://github.com/wojciechz/learning_to_execute) code from Wojciech Zaremba.

