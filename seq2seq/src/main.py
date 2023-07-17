# %%
import sys
import time
import math
import logging
import torch
from torch import nn, optim
# from seq2seq.src.data_set import DataSets
from seq2seq.src.data_loader import data_loader
from seq2seq.src.trainer import train, evaluate, epoch_time
from seq2seq.src import models
from seq2seq.src import utils as u


logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

# # %%
# train_datasets = DataSets(mode="train")
# val_datasets = DataSets(mode="val")
# test_datasets = DataSets(mode="test")

# %%
output_dir = "/app/seq2seq/data/outputs/"
train_filepath = output_dir + "train_datasets.pkl"
val_filepath = output_dir + "val_datasets.pkl"
test_filepath = output_dir + "test_datasets.pkl"

# # %%
# u.save_pickle(train_filepath, train_datasets)
# u.save_pickle(val_filepath, val_datasets)
# u.save_pickle(test_filepath, test_datasets)

# %%
train_datasets = u.load_pickle(train_filepath)
val_datasets = u.load_pickle(val_filepath)
test_datasets = u.load_pickle(test_filepath)

# %%
logger.info(train_datasets.de_vocab[:10])
logger.info(train_datasets.en_vocab[:10])

# %%
BATCH_SIZE = 5
train_loader = data_loader(train_datasets, batch_size=BATCH_SIZE)
val_loader = data_loader(val_datasets, batch_size=BATCH_SIZE)
test_loader = data_loader(test_datasets, batch_size=BATCH_SIZE)


# %%
INPUT_DIM = train_datasets.n_de_vocab
OUTPUT_DIM = train_datasets.n_en_vocab
logger.info("input_dim: %s", INPUT_DIM)
logger.info("output_dim: %s", OUTPUT_DIM)


ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info("device: %s", device)

# %%
enc = models.Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
attn = models.Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
dec = models.Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
model = models.Seq2Seq(enc, dec, device).to(device)


# %%
def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)
optimizer = optim.Adam(model.parameters())


def count_parameters(model_: nn.Module):
    return sum(p.numel() for p in model_.parameters() if p.requires_grad)


logger.info('The model has %s trainable parameters', count_parameters(model))


# %%
N_EPOCHS = 1
CLIP = 1

best_valid_loss = float('inf')
PAD_IDX = train_datasets.pad_idx
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, val_loader, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

test_loss = evaluate(model, test_loader, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
