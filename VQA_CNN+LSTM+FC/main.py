import torch
import torch.nn as nn
import torchtext
import os
import random
import numpy as np
import pandas as pd
import spacy
import timm
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchvision import transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




# Load the English tokenizer from spaCy
eng = spacy.load("en_core_web_sm")

def get_tokens(data_iter):
    for sample in data_iter:
        question = sample['question']
        yield [token.text for token in eng.tokenizer(question)]


def tokenize(question, max_seq_len):
    tokens = [token.text for token in eng.tokenizer(question)]
    sequence = [vocab[token] for token in tokens]
    if len(sequence) < max_seq_len:
        sequence += [vocab['<pad>']] * (max_seq_len - len(sequence))
    else:
        sequence = sequence[:max_seq_len]

    return sequence

class VQADataset(Dataset):
    def __init__(
        self,
        data,
        label2idx,
        max_seq_len=20,
        transform=None,
        img_dir='/home/always/VQA/Project-Visual-Question-Answering/data/vqa_coco_dataset/val2014-resised/'
    ):
        self.transform = transform
        self.data = data
        self.max_seq_len = max_seq_len
        self.img_dir = img_dir
        self.label2idx = label2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.data[index]['image_path'])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        question = self.data[index]['question']
        question = tokenize(question, self.max_seq_len)
        question = torch.tensor(question, dtype=torch.long)

        label = self.data[index]['answer']
        label = self.label2idx[label]
        label = torch.tensor(label, dtype=torch.long)

        return img, question, label
class VQAModel(nn.Module):
    def __init__(
        self,
        n_classes,
        img_model_name,
        embeddding_dim,
        n_layers=2,
        hidden_size=256,
        drop_p=0.2
    ):
        super(VQAModel, self).__init__()
        
        # Image encoder
        self.image_encoder = timm.create_model(
            img_model_name,
            pretrained=True,
            num_classes=hidden_size
        )

        # Fine-tune image encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = True

        # Text embedding and LSTM layers
        self.embedding = nn.Embedding(len(vocab), embeddding_dim)
        self.lstm1 = nn.LSTM(
            input_size=embeddding_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=drop_p
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 3, hidden_size)
        self.dropout = nn.Dropout(drop_p)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, img, text):
        # Encode image features
        img_features = self.image_encoder(img)

        # Encode text features
        text_emb = self.embedding(text)
        lstm_out, _ = self.lstm1(text_emb)

        # Use the last hidden state from LSTM
        lstm_out = lstm_out[:, -1, :]

        # Combine image and text features
        combined = torch.cat((img_features, lstm_out), dim=1)

        # Fully connected layers
        x = self.fc1(combined)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    losses = []

    with torch.no_grad():
        for image, question, labels in dataloader:
            image, question, labels = image.to(device), question.to(device), labels.to(device)
            outputs = model(image, question)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = sum(losses) / len(losses)
    acc = correct / total

    return loss, acc

def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs
):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        batch_train_losses = []

        model.train()
        for idx, (images, questions, labels) in enumerate(train_loader):
            images, questions, labels = images.to(device), questions.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, questions)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_train_losses.append(loss.item())

        train_loss = sum(batch_train_losses) / len(batch_train_losses)
        train_losses.append(train_loss)

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)

        print(f'EPOCH {epoch + 1}: Train loss: {train_loss:.4f} Val loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')

        scheduler.step()

    return train_losses, val_losses
if __name__ == '__main__':

    seed = 59
    set_seed(seed)
    # Load train data
    train_data = []
    train_set_path = '/home/always/VQA/Project-Visual-Question-Answering/data/vqa_coco_dataset/vaq2.0.TrainImages.txt'

    with open(train_set_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split('\t')
            qa = temp[1].split('?')

            if len(qa) == 3:
                answer = qa[2].strip()
            else:
                answer = qa[1].strip()

            data_sample = {
                'image_path': temp[0][:-2],
                'question': qa[0] + '?',
                'answer': answer
            }
            train_data.append(data_sample)

    # Load val data
    val_data = []
    val_set_path = '/home/always/VQA/Project-Visual-Question-Answering/data/vqa_coco_dataset/vaq2.0.DevImages.txt'

    with open(val_set_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split('\t')
            qa = temp[1].split('?')

            if len(qa) == 3:
                answer = qa[2].strip()
            else:
                answer = qa[1].strip()

            data_sample = {
                'image_path': temp[0][:-2],
                'question': qa[0] + '?',
                'answer': answer
            }
            val_data.append(data_sample)

    # Load test data
    test_data = []
    test_set_path = '/home/always/VQA/Project-Visual-Question-Answering/data/vqa_coco_dataset/vaq2.0.TestImages.txt'

    with open(test_set_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split('\t')
            qa = temp[1].split('?')

            if len(qa) == 3:
                answer = qa[2].strip()
            else:
                answer = qa[1].strip()

            data_sample = {
                'image_path': temp[0][:-2],
                'question': qa[0] + '?',
                'answer': answer
            }
            test_data.append(data_sample)


    # Build the vocabulary from the training data tokens
    vocab = build_vocab_from_iterator(
        get_tokens(train_data),
        min_freq=2,
        specials=['<pad>', '<sos>', '<eos>', '<unk>'],
        special_first=True
    )

    # Set the default index for unknown tokens
    vocab.set_default_index(vocab['<unk>'])


    # Create label mapping
    classes = set([sample['answer'] for sample in train_data])
    label2idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    idx2label = {idx: cls_name for idx, cls_name in enumerate(classes)}



    data_transform = {
    'train': transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.CenterCrop(size=180),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
}
    train_dataset = VQADataset(
    train_data,
    label2idx=label2idx,
    transform=data_transform['train']
)

    val_dataset = VQADataset(
        val_data,
        label2idx=label2idx,
        transform=data_transform['val']
    )

    test_dataset = VQADataset(
        test_data,
        label2idx=label2idx,
        transform=data_transform['val']
    )
    train_batch_size = 256
    test_batch_size = 32

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )

    n_classes = len(classes)
    img_model_name = 'resnet18'
    hidden_size = 256
    n_layers = 2
    embeddding_dim = 128
    drop_p = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = VQAModel(
        n_classes=n_classes,
        img_model_name=img_model_name,
        embeddding_dim=embeddding_dim,
        n_layers=n_layers,
        hidden_size=hidden_size,
        drop_p=drop_p
    ).to(device)

    lr = 1e-3
    epochs = 50

    scheduler_step_size = int(epochs * 0.8)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=0.1
    )

    train_losses, val_losses = fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs
)
    
    val_loss, val_acc = evaluate(
    model,
    val_loader,
    criterion,
    device
)

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print('Evaluation on val/test dataset')
    print('Val accuracy:', val_acc)
    print('Test accuracy:', test_acc)
