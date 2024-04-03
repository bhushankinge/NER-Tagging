import pandas as pd
import itertools
import numpy as np
import os
import itertools
from collections import Counter
from datasets import Dataset, DatasetDict

label_mapping = {
    'O': 0,
    'B-ORG': 1,
    'I-ORG': 2,
    'B-PER': 3,
    'I-PER': 4,
    'B-LOC': 5,
    'I-LOC': 6,
    'B-MISC': 7,
    'I-MISC': 8
}

def file_to_dataset_for_test(file_path):
    tokens = []
    data = {'id': [], 'tokens': []}
    current_id = 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # If the line is not empty, process it
                parts = line.strip().split(' ')
                if len(parts) == 2:  # Ensure the line has three parts: index, word
                    _, token = parts
                    tokens.append(token)
            else:  # If the line is empty, it indicates the end of a sentence
                if tokens:  # If there are tokens collected, save the current sentence
                    data['id'].append(str(current_id))
                    data['tokens'].append(tokens)
                    current_id += 1
                    tokens = []
                
        # Add the last sentence if the file doesn't end with a blank line
        if tokens:
            data['id'].append(str(current_id))
            data['tokens'].append(tokens)
    
    dataset = Dataset.from_dict(data)
    return dataset

test_dataset = file_to_dataset_for_test('data/test')

def file_to_dataset(file_path, label_mapping):
    tokens = []
    labels = []
    data = {'id': [], 'tokens': [], 'labels': []}
    current_id = 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # If the line is not empty, process it
                parts = line.strip().split(' ')
                if len(parts) == 3:  # Ensure the line has three parts: index, word, NER tag
                    _, token, label = parts
                    tokens.append(token)
                    labels.append(label_mapping[label])  # Convert label to integer
            else:  # If the line is empty, it indicates the end of a sentence
                if tokens:  # If there are tokens collected, save the current sentence
                    data['id'].append(str(current_id))
                    data['tokens'].append(tokens)
                    data['labels'].append(labels)
                    current_id += 1
                    tokens, labels = [], []  # Reset for the next sentence
                
        # Add the last sentence if the file doesn't end with a blank line
        if tokens:
            data['id'].append(str(current_id))
            data['tokens'].append(tokens)
            data['labels'].append(labels)
    
    dataset = Dataset.from_dict(data)
    return dataset

train_path = 'data/train'
validation_path = 'data/dev'

# Converting files to Hugging Face Datasets
train_dataset = file_to_dataset(train_path, label_mapping)
validation_dataset = file_to_dataset(validation_path, label_mapping)

# Combining datasets into a DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset
})

tag2idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
idx2tag = {v:k for k, v in tag2idx.items()}

from itertools import chain
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import itertools
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def calculate_metrics(labels, preds):
    # Flatten the lists
    flat_labels = list(chain(*labels))
    flat_preds = list(chain(*preds))

    # Initialize counts
    TP = FP = FN = 0

    # Calculate TP, FP, FN
    for true_label, pred_label in zip(flat_labels, flat_preds):
        if true_label != 'O':  # Entity present in the ground truth
            if true_label == pred_label:
                TP += 1  # Correctly identified entity
            else:
                FN += 1  # Missed entity
        if pred_label != 'O':  # Entity predicted
            if true_label != pred_label:
                FP += 1  # Incorrectly identified entity

    # Calculate precision, recall, and F1 score
    precision = (TP / (TP + FP) if (TP + FP) else 0) * 100
    recall = (TP / (TP + FN) if (TP + FN) else 0) * 100
    f1 = (2 * (precision * recall) / (precision + recall)) if (precision + recall) else 0

    return precision, recall, f1

d = {'preds': [], 'labels': [], 'mask': [], 'loss': []}
def eval_dataloader(loader, model, loss_fn, verbose=False, name=''):
    
    # using cuda only as required. Only GPU
    for batch in loader:
        input_ids = batch['input_ids'].to('cuda')
        labels = batch['labels'].to('cuda')
        mask = batch['mask'].to('cuda')
        
        predictions = model(input_ids)
        # print(predictions[0])
        loss = loss_fn(predictions.transpose(-1, -2), labels.to(torch.long))
        loss = torch.masked_select(loss, mask.bool()).mean()
        
        d['preds'].extend(predictions.argmax(-1).tolist())
        d['labels'].extend(labels.tolist())
        d['mask'].extend(mask.tolist())
        d['loss'].append(loss.item())
        
    preds, labels = [], []
    # print(d['preds'])
    # create eval fun that support data
    for i in range(len(d['preds'])):
        pred = [idx2tag[k] for k, m in zip(d['preds'][i], d['mask'][i]) if m > 0]
        label = [idx2tag[k] for k, m in zip(d['labels'][i], d['mask'][i]) if m > 0]
        preds.append(pred)
        labels.append(label)

    
    precision, recall, f1 = calculate_metrics(labels, preds)
    # print/return the average loss and f1
    print(f'name: {name}, f1: {f1}, precision: {precision}, recall: {recall}, loss: {sum(d["loss"])/len(d["loss"])}')
    return (f1, precision, recall), sum(d['loss'])/len(d['loss'])

# iterate on tokens and count using Counter class
word_frequency = Counter(itertools.chain(*dataset['train']['tokens']))
word2idx = {
    word: frequency
    for word, frequency in word_frequency.items()
    if frequency >= 3
}

word2idx = {
    word: index
    for index, word in enumerate(word2idx.keys(), start=2)
}
print('vocab count', len(word2idx))

word2idx['[PAD]'] = 0
word2idx['[UNK]'] = 1
idx2word = {v:k for k, v in word2idx.items()}

def convert_word_to_id(sample):
    return {
        'input_ids': [
        (word2idx[token] if token in word2idx else word2idx['[UNK]'])
        for token in sample['tokens']
        ]
}
dataset_ids = dataset.map(convert_word_to_id)
test_dataset_ids = test_dataset.map(convert_word_to_id)

class MaxLenFinder:
    def __init__(self):
        self.max_len = 0

    def get_max_len(self, sample):
        if len(sample['tokens']) > self.max_len: 
            self.max_len = len(sample['tokens'])
        # It's important for the map function that we return the sample unchanged.
        return sample

# Usage for dataset 1
max_len_finder = MaxLenFinder()
dataset.map(max_len_finder.get_max_len)
train_max_len = max_len_finder.max_len

test_len_finer = MaxLenFinder()
test_dataset.map(max_len_finder.get_max_len)
test_max_len = max_len_finder.max_len


max_len = max(train_max_len, test_max_len)

# dataset class. NERDataset
class NER_Dataset_test(Dataset):
    def __init__(self, dataset, vocab, max_len=128):
        self.dataset = dataset
        self.vocab = vocab
        self.max_len = max_len
    def __len__(self):
        return self.dataset.num_rows
    def __getitem__(self, idx):
        data = self.dataset[idx]
        input_ids = data['input_ids']
        mask = [1] * len(data['input_ids'])
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.int32),
            'mask': torch.tensor(mask, dtype=torch.int8)
        }

NERtest_dataset = NER_Dataset_test(test_dataset_ids, word2idx)

# dataset class. NERDataset
class NER_Dataset(Dataset):
    def __init__(self, dataset, vocab, max_len=128):
        self.dataset = dataset
        self.vocab = vocab
        self.max_len = max_len
    def __len__(self):
        return self.dataset.num_rows
    def __getitem__(self, idx):
        data = self.dataset[idx]
        input_ids = data['input_ids']
        label = data['labels']
        mask = [1] * len(data['input_ids'])
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.int32),
            'labels': torch.tensor(label, dtype=torch.int8),
            'mask': torch.tensor(mask, dtype=torch.int8)
        }
    
# create datasets. Train, validation, test
train_dataset = NER_Dataset(dataset_ids['train'], word2idx)
val_dataset = NER_Dataset(dataset_ids['validation'], word2idx)

# define collate function. Passing the collate function can inputs
def collate_fn(inputs, pad_token_id = 0):
    return {
        'input_ids': nn.utils.rnn.pad_sequence([i['input_ids'] for i in inputs], batch_first=True, padding_value=pad_token_id),
        'labels': nn.utils.rnn.pad_sequence([i['labels'] for i in inputs], batch_first=True, padding_value=pad_token_id),
        'mask': nn.utils.rnn.pad_sequence([i['mask'] for i in inputs], batch_first=True, padding_value=pad_token_id),
    }


def collate_fn_test(batch):
    input_ids = [item['input_ids'] for item in batch]
    masks = [item['mask'] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=0)

    return {'input_ids': input_ids_padded, 'mask': masks_padded}

# create data loader
batch_size=32
train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size)
test_loader = DataLoader(NERtest_dataset, collate_fn=collate_fn_test, batch_size=batch_size)

class BiLSTM(nn.Module):
        def __init__(self, input_dim, output_dim, embed_dim=100, lstm_dim=256, dropout=0.50, linear_dim=128):
                super().__init__()
                self.input_dim = input_dim
                self.embed_dim = embed_dim
                self.embedding = nn.Embedding(self.input_dim, self.embed_dim)
                self.lstm_dim = lstm_dim
                self.lstm = nn.LSTM(self.embed_dim, self.lstm_dim, batch_first=True, bidirectional=True)
                self.dropout = nn.Dropout(dropout)
                self.linear_dim = linear_dim
                self.fc2 = nn.Linear(self.lstm_dim * 2, self.linear_dim)
                self.elu = nn.ELU()
                self.output_dim = output_dim
                self.fc3 = nn.Linear(self.linear_dim, self.output_dim)
            
        def forward(self, input_ids):
                batch_size = input_ids.shape[0]
                emds = self.embedding(input_ids)
                h0 = torch.randn(2, batch_size, self.lstm_dim).to(input_ids.device)
                c0 = torch.randn(2, batch_size, self.lstm_dim).to(input_ids.device)
                output, (hn, cn) = self.lstm(emds, (h0, c0))
                output = self.dropout(output)
                linear_out = self.fc2(output)
                linear_out = self.dropout(linear_out)
                elu_out = self.elu(linear_out)
                results = self.fc3(elu_out)
                return results
        


bilstm_model = BiLSTM(input_dim=len(word2idx), output_dim=len(tag2idx)).to(device)
def create_hyperparameters(model, epochs):
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-8)
        num_epochs = epochs
        return loss_fn, optimizer, scheduler, num_epochs

loss_fn, optimizer, scheduler, num_epochs = create_hyperparameters(bilstm_model, 10)
eval_dataloader(val_loader, bilstm_model, loss_fn=loss_fn, name='test')

def training_loop(loss_fn, optimizer, scheduler, num_epochs, model, train_loader):
        losses = []
        vlosses = []
        lrs = []
        for epoch in range(num_epochs):
                model.train()
                total_loss = 0.0
                loss = 0.0
                pbar = tqdm(train_loader)
                for batch in pbar:
                        input_ids = batch['input_ids'].to(device)
                        labels = batch['labels'].to(device)
                        mask = batch['mask'].to(device)
                        optimizer.zero_grad()
                        predictions = model(input_ids)
                        loss = loss_fn(predictions.transpose(-1, -2), labels.to(torch.long))
                        loss = torch.masked_select(loss, mask.bool()).mean()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        pbar.set_postfix({'loss': loss.item()})
                lrs.append(scheduler.get_last_lr())
                scheduler.step()
                average_loss = total_loss / len(train_loader)
                print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}')
                model.eval()
                f1, l = eval_dataloader(train_loader, model, loss_fn=loss_fn, name='train')
                losses.append(l)
        return losses, vlosses, lrs


losses, vlosses, lrs = training_loop(*create_hyperparameters(bilstm_model, 25), bilstm_model, train_loader)

models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

# It's good practice to include the '.pt' extension for PyTorch model files
model_path = os.path.join(models_dir, 'blstm1.pt')  # Updated to include the .pt extension

# Save the model state dictionary, overwriting any existing file
torch.save(bilstm_model.state_dict(), model_path)

print(f"Model saved to {model_path}")

# F1, precision, and recall result for eval
f1 = eval_dataloader(val_loader, bilstm_model.eval(), loss_fn=loss_fn, name='Dev', verbose=True)

def write_predictions_to_file(dataloader, model, idx2tag, file_path):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad(), open(file_path, 'w', encoding='utf-8') as file:
        for batch in dataloader:
            input_ids = batch['input_ids'].to('cuda')
            # No labels are needed, as we're just predicting
            mask = batch['mask'].to('cuda')

            predictions = model(input_ids)
            predictions = predictions.argmax(-1)  # Get the most likely prediction indices

            # Convert the batch of predictions to tags
            for sentence_preds, sentence_mask in zip(predictions, mask):
                idx = 1  # Start index for each new sentence
                for pred, m in zip(sentence_preds, sentence_mask):
                    if m == 0:  # Skip padding
                        continue
                    tag = idx2tag[int(pred)]
                    
                    file.write(f"{idx} TOKEN {tag}\n")
                    idx += 1
                file.write("\n")  # Separate sentences by a blank line


def process_predictions(loader, model, idx2tag, dataset_partition, output_file, temp_file='temp_predictions.txt'):
    """
    Args:
    - loader: DataLoader for the dataset.
    - model: The model to make predictions.
    - idx2tag: Dictionary to map prediction indices to tags.
    - dataset_partition: Part of the dataset to use (e.g., 'validation', 'test').
    - output_file: File path for the output file.
    """
    # Step 1: Write predictions to a temporary file
    write_predictions_to_file(loader, model, idx2tag, temp_file)

    # Step 2: Read tokens from the dataset
    token_lists = [dataset_partition[i]['tokens'] for i in range(len(dataset_partition))]

    # Step 3: Read and process the temporary file content
    with open(temp_file, 'r') as file:
        content = file.read()

    blocks = content.strip().split('\n\n')
    assert len(blocks) == len(token_lists), "The number of blocks and token lists do not match."

    processed_blocks = []
    for block, tokens in zip(blocks, token_lists):
        lines = block.split('\n')
        new_lines = []
        for line, token in zip(lines, tokens):
            new_line = line.replace('TOKEN', token)
            new_lines.append(new_line)
        processed_blocks.append('\n'.join(new_lines))

    output_content = '\n\n'.join(processed_blocks)

    # Step 4: Write the processed content to the specified output file
    with open(output_file, 'w') as file:
        file.write(output_content)

    # Step 5: Clean up the temporary file
    if os.path.exists(temp_file):
        os.remove(temp_file)


from torch.nn.utils.rnn import pad_sequence

process_predictions(val_loader, bilstm_model, idx2tag, dataset['validation'], 'dev1.out')
process_predictions(test_loader, bilstm_model, idx2tag, test_dataset, 'test1.out')


glove_path = 'glove.6B.100d.txt'
glove = {}
with open(glove_path, 'r', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        glove[word] = [-0.1, *vector]
        if word[0].upper() != word[0]:
            glove[word[0].upper()+word[1:]] = [0.1, *vector]
            if len(word) > 1:
                glove[word.upper()] = [0.2, *vector]


pu_emb = torch.zeros(2, 101)
pu_emb.shape

glove_tensored = torch.tensor(list(glove.values()))
glove_tensored.mean(), glove_tensored.std()

glove_embeddings = torch.cat([pu_emb, glove_tensored], dim=0)

gword2idx = {
    word: index
    for index, word in enumerate(glove.keys(), start=2)
}
gword2idx['[PAD]'] = 0
gword2idx['[UNK]'] = 1


# Convert all tokens to their respective indx
def gconvert_word_to_id(sample):
    return {
        'input_ids': [
            (gword2idx[token] if token in gword2idx else gword2idx['[UNK]'])
            for token in sample['tokens']
    ]
}
gdataset_ids = dataset.map(gconvert_word_to_id)

test_gdataset_ids = test_dataset.map(gconvert_word_to_id)

gtrain_dataset = NER_Dataset(gdataset_ids['train'], gword2idx)
gval_dataset = NER_Dataset(gdataset_ids['validation'], gword2idx)
gtest_dataset = NER_Dataset_test(test_gdataset_ids, gword2idx)


# create data loader
gtrain_loader = DataLoader(gtrain_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
gval_loader = DataLoader(gval_dataset, collate_fn=collate_fn, batch_size=batch_size)
gtest_loader = DataLoader(gtest_dataset, collate_fn=collate_fn_test, batch_size=batch_size)

class GloveBiLSTM(nn.Module):
    def __init__(self, output_dim, glove_embeddings, embed_dim=101, lstm_dim=256,
                dropout=0.50, linear_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding.from_pretrained(glove_embeddings, freeze=True)
        self.lstm_dim = lstm_dim
        self.lstm = nn.LSTM(self.embed_dim, self.lstm_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_dim = linear_dim
        self.fc2 = nn.Linear(self.lstm_dim * 2, self.linear_dim)
        self.elu = nn.ELU()
        self.output_dim = output_dim
        self.fc3 = nn.Linear(self.linear_dim, self.output_dim)

    def forward(self, input_ids):
        batch_size = input_ids.shape[0]
        emds = self.embedding(input_ids)
        h0 = torch.randn(2, batch_size, self.lstm_dim).to(input_ids.device)
        c0 = torch.randn(2, batch_size, self.lstm_dim).to(input_ids.device)
        output, (hn, cn) = self.lstm(emds, (h0, c0))
        output = self.dropout(output)
        linear_out = self.fc2(output)
        linear_out = self.dropout(linear_out)
        elu_out = self.elu(linear_out)
        results = self.fc3(elu_out)
        return results

# train model
gbilstm_model = GloveBiLSTM(output_dim=len(tag2idx), glove_embeddings=glove_embeddings).to(device)
losses, vlosses, lrs = training_loop(*create_hyperparameters(gbilstm_model, 25), gbilstm_model, gtrain_loader)

models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

# Save the model state dictionary, overwriting any existing file
model_path = os.path.join(models_dir, 'blstm2.pt')
torch.save(gbilstm_model.state_dict(), model_path)

print(f"Model saved to {model_path}")

# F1, precision, and recall result for eval
f1 = eval_dataloader(gval_loader, gbilstm_model.eval(), loss_fn=loss_fn, name='Dev', verbose=True)


process_predictions(gval_loader, gbilstm_model, idx2tag, dataset['validation'], 'dev2.out')
process_predictions(gtest_loader, gbilstm_model, idx2tag, test_dataset, 'test2.out')
