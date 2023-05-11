from main import Transformer
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

# Define the fields for the input and output sentences
SRC = Field(tokenize="spacy", tokenizer_language="fr", init_token="<sos>", eos_token="<eos>", lower=True)
TRG = Field(tokenize="spacy", tokenizer_language="en", init_token="<sos>", eos_token="<eos>", lower=True)

# Load the data
train_data, valid_data, test_data = Multi30k.splits(exts=(".fr", ".en"), fields=(SRC, TRG))
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# Define the model
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HIDDEN_DIM = 256
NUM_LAYERS = 3
NUM_HEADS = 8

model = Transformer(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi[TRG.pad_token])

# Define the training loop
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    
    for batch in iterator:
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg[:, :-1])
        
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

# Train the model
BATCH_SIZE = 32
N_EPOCHS = 10

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    datasets=(train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
)

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f}")

# Test the model
def translate_sentence(model, sentence, src_field, trg_field, max_len=50):
    model.eval()
    
    if isinstance(sentence, str):
        tokens = [token.text.lower() for token in spacy_fr(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0)
    
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        encoder_outputs = model.encoder(src_tensor, src_mask)
    
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0)
        
        trg_mask = model.make_trg_mask(trg_tensor)
        print(trg_mask)