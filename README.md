# Handwriting Synthesis

## About
A from-scratch TensorFlow implementation of the handwriting synthesis network from:

> Alex Graves, "Generating Sequences With Recurrent Neural Networks" (2013). [arXiv:1308.0850](https://arxiv.org/abs/1308.0850)

Given a text string, the model generates a plausible pen-stroke sequence for handwriting
it out, using a stack of 3 LSTM layers with peephole connections, a Gaussian-window soft
attention mechanism over the input text, and a mixture density network (20-component
bivariate Gaussian mixture + Bernoulli pen-lift) output layer.

## Dataset
This project uses the IAM On-Line Handwriting Database (IAM-OnDB), which must be downloaded separately after registering on the FKI website:

> M. Liwicki and H. Bunke, "IAM-OnDB - an On-Line English Sentence Database Acquired from Handwritten Text on a Whiteboard." 8th Intl. Conf. on Document Analysis and Recognition (2005), Volume 2, pp. 956-961.

The database must be placed in a folder called Dataset, and the stroke sequences in Dataset/Strokes/lineStrokes

## Setup
Requires Python 3.13.
```
py -3.13 -m venv venv
```
Activate (pick the one for your shell):
```
venv\Scripts\activate          # cmd.exe
venv\Scripts\Activate.ps1      # PowerShell
source venv/Scripts/activate   # Git Bash -- must be "sourced", not run directly
```
Then install dependencies:
```
pip install -r requirements.txt
```

## Run
Activate the venv (see above), then:
```
python Training.py
```

Generate handwriting from text with a trained checkpoint:
```
python Testing.py "text to write"
```

## Results
Training progress, epoch 0 to 67 (sampled every epoch at batch 100):

![Training progress](Assets/training_progress.gif)

Validation loss over training, with a linear trend line:

![Validation loss](Assets/validation_loss.png)

## Implementation notes
The recurrent core (3 peephole LSTMs + attention window) originally ran fully eager and
padded every batch out to the dataset's longest sequence (1940 timesteps) regardless of a
given batch's actual content length -- 70-220+ seconds per batch. It's now a single
`tf.while_loop` compiled under `@tf.function`, with each batch dynamically trimmed to its
own real length inside the traced graph (so the trim itself doesn't force a retrace). That
took it down to 2-5 seconds per batch, a 30-80x speedup, which is what actually made
training past epoch 20 practical on CPU.

The model's equations (peephole LSTM gates, the soft attention window, the mixture density
loss, gradient clipping ranges, the RMSprop variant) were checked line by line against the
paper. One deliberate deviation: the paper's best-reported results use a second-stage
"adaptive weight noise" fine-tune, which isn't implemented here -- what's here is the base
model the paper itself trains before that refinement step.