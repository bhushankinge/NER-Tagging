# Named Entity Recognition using BiLSTM with Random and GloVe Embeddings

This repository contains code for Named Entity Recognition (NER) using Bidirectional LSTM (BiLSTM) models. The code implements two approaches: one using randomly initialized word embeddings and another using pre-trained GloVe word embeddings. The NER task involves identifying and classifying named entities such as persons, organizations, locations, and miscellaneous entities in text data.

## Data

The data is provided in a specific format, where each line represents a single token, and sentences are separated by an empty line. The data should be in the following format:

```
<word_index> <token> <label>
1 EU B-ORG
2 rejects O
3 German B-MISC
4 call O
5 to O
...
```

The labels should be in the IOB (Inside, Outside, Beginning) format, such as `B-ORG` (Beginning of an Organization), `I-ORG` (Inside an Organization), `B-PER` (Beginning of a Person), `I-PER` (Inside a Person), `B-LOC` (Beginning of a Location), `I-LOC` (Inside a Location), `B-MISC` (Beginning of a Miscellaneous entity), `I-MISC` (Inside a Miscellaneous entity), and `O` (Outside of any entity).

## Usage

1. Place your train, dev (validation), and test data files in the `data` directory.
2. If using the GloVe embeddings, download the GloVe file (`glove.6B.100d.txt`) and place it in the project directory.
3. Run the code, which will preprocess the data, create datasets, and train the BiLSTM models.
4. After training, the code will save the trained models in the `models` directory with the filenames `blstm1.pt` (randomly initialized embeddings) and `blstm2.pt` (GloVe embeddings).
5. The code will also generate prediction files for the dev and test sets, named `dev1.out`, `test1.out`, `dev2.out`, and `test2.out`.

## Dependencies

The code requires the following Python packages:

- pandas
- numpy
- datasets
- torch
- tqdm

You can install these dependencies using pip:

```
pip install pandas numpy datasets torch tqdm
```
or, you can just download and install the requirements.txt file.

## Models

The code defines two PyTorch modules for BiLSTM models:

1. `BiLSTM`: This model uses randomly initialized word embeddings.
2. `GloveBiLSTM`: This model uses pre-trained GloVe word embeddings.

Both models take word embeddings as input and produce a sequence of predictions, one for each token in the input sequence.

## Training

The training process is handled by the `training_loop` function, which performs the following steps:

1. Iterate over the specified number of epochs.
2. For each batch in the training data loader:
   - Forward pass through the model.
   - Compute the loss using CrossEntropyLoss.
   - Backward pass and update the model parameters.
3. Evaluate the model on the validation set after each epoch.
4. Save the model after training is complete.

## Evaluation

The code includes functions for evaluating the model's performance on the validation and test sets:

- `eval_dataloader`: Computes the loss and F1, precision, and recall scores for a given data loader.
- `write_predictions_to_file`: Writes the model's predictions to a file in the expected format.
- `process_predictions`: Processes the predictions by combining them with the original token data and writes the results to a specified output file.

The `process_predictions` function is used to generate the `dev1.out`, `test1.out`, `dev2.out`, and `test2.out` files containing the models' predictions on the dev and test sets.

Note: The code assumes that the data is in the specified format and that the necessary directories (`data` and `models`) exist in the repository. You may need to modify the code or provide additional instructions if your data or directory structure differs.

## Contributing

Feel free to fork the repository and submit pull requests.
