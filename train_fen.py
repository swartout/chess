# imports
from typing import List, Dict
import itertools

from fenparser import FenParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from transformers import BertModel, BertTokenizer

from tqdm import tqdm


# ---------------------------------- config ------------------------------------

FEN_FILE = 'fens.txt'
BATCH_SIZE = 32
LR = 3e-4
ITERATIONS = 100
FEN_LEN = 64
BOARD_LEN = 768
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
SAVE_PATH = 'models/fenmodel.pt'

# ---------------------------------- config ------------------------------------


def fen_to_targets(f: str) -> torch.Tensor:
    p = FenParser(f)
    b = list(itertools.chain.from_iterable(p.parse()))
    pieces = ['p', 'r', 'n', 'b', 'q', 'k', 'P', 'R', 'N', 'B', 'Q', 'K']
    out = torch.zeros(BOARD_LEN)
    for i in range(12):
        target = torch.zeros(FEN_LEN)
        target[[pieces[i] == x for x in b]] = 1
        out[FEN_LEN*i:FEN_LEN*(i+1)] = target
    return out


class FenDataset(Dataset):
    def __init__(self, fens: List[str], tokenizer: BertTokenizer, max_len: int = 64):
        self.fens = fens
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.fens)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer.encode_plus(
            self.fens[idx],
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True
        )
        targets = fen_to_targets(self.fens[idx])
        return {
            'input_ids': torch.tensor(inputs['input_ids']),
            'attention_mask': torch.tensor(inputs['attention_mask']),
            'token_type_ids': torch.tensor(inputs['token_type_ids']),
            'targets': targets
        }


class FenModel(nn.Module):
    def __init__(self, out_size: int = BOARD_LEN, dropout: bool = False):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.bert.pooler = nn.Linear(BOARD_LEN, BOARD_LEN)
        self.linear = nn.Linear(BOARD_LEN, out_size)
        if not dropout:
            self.dropout = nn.Dropout(0.2)
        else:
            self.dropout = None

    def forward(self, x: Dict[str, torch.Tensor]):
        x = self.bert(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            token_type_ids=x['token_type_ids'],
            return_dict=False
        )[0][:,0]
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear(x)
        return x


def train_model(model, optimizer, dataloader, iterations, device='cpu'):
    train_losses = []
    train_acc = []
    
    for i, data in tqdm(enumerate(dataloader), total=iterations):
        model.train()
        data = {k: v.to(device) for k, v in data.items()}
        optimizer.zero_grad()
        labels_pred = model(data)
        batch_loss = F.binary_cross_entropy_with_logits(labels_pred, data['targets'])
    
        batch_loss.backward()
        optimizer.step()
    
        train_losses.append(batch_loss.item())
    
        labels_pred_binary = torch.zeros_like(data['targets'])
        labels_pred_binary[labels_pred > 0] = 1.0
        train_acc.append(torch.mean((labels_pred_binary == data['targets']).float()).item())
    
        if (i+1) % (iterations // 20) == 0:
            print(f'Iteration: {i+1}, train_loss: {train_losses[-1]:.4f}, train_acc: {train_acc[-1]:.4f}')

        if i == iterations:
            return train_losses, train_acc


def get_val(model, dataloader, iterations, device='cpu'):
    val_losses = []
    val_acc = []
    
    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=iterations):
            data = {k: v.to(device) for k, v in data.items()}
            labels_pred = model(data)
            batch_loss = F.binary_cross_entropy_with_logits(labels_pred, data['targets'])
            
            val_losses.append(batch_loss.item())
            labels_pred_binary = torch.zeros_like(data['targets'])
            labels_pred_binary[labels_pred > 0] = 1.0
            val_acc.append(torch.mean((labels_pred_binary == data['targets']).float()).item())
            
            if i == iterations:
                return val_losses, val_acc


def main():
    # print config parameters
    print('Running training script with the following parameters:')
    print(f'FEN_FILE: {FEN_FILE}')
    print(f'BATCH_SIZE: {BATCH_SIZE}')
    print(f'LR: {LR}')
    print(f'ITERATIONS: {ITERATIONS}')
    print(f'FEN_LEN: {FEN_LEN}')
    print(f'BOARD_LEN: {BOARD_LEN}')
    print(f'DEVICE: {DEVICE}')
    print(f'SAVE_PATH: {SAVE_PATH}')

    # load data
    print('Loading data')
    with open(FEN_FILE, 'r') as f:
        fens = [row for row in f]

    # create datasets
    print('Creating datasets')
    split = int(len(fens) * 0.9)
    train_split = fens[:split]
    test_split = fens[split:]
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    train_ds = FenDataset(train_split, tokenizer, FEN_LEN)
    val_ds = FenDataset(test_split, tokenizer, FEN_LEN)

    # create dataloader
    print('Creating dataloaders')
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)

    # create model
    print('Creating model')
    model = FenModel().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)

    # train model
    train_losses, train_acc = train_model(model, optimizer, train_dl, ITERATIONS, device=DEVICE)

    # get validation stats
    val_losses, val_acc = get_val(model, val_dl, ITERATIONS // 10, device=DEVICE)
    print(f'Validation loss: {torch.mean(torch.tensor(val_losses)).item():.4f}')
    print(f'Validation acc:  {torch.mean(torch.tensor(val_acc)).item():.4f}')

    # save model
    torch.save(model.state_dict(), SAVE_PATH)
    print(f'Saved model to: {SAVE_PATH}')

if __name__ == '__main__':
    main()
