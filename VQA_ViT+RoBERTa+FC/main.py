import torch
import torch.nn as nn
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel, ViTImageProcessor
from transformers import AutoTokenizer, RobertaModel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class VQADataset(Dataset):
    def __init__(
        self,
        data,
        label2idx,
        img_feature_extractor,
        text_tokenizer,
        device,
        transforms=None,
        img_dir='val2014-resised'
    ):
        self.data = data
        self.img_dir = img_dir
        self.label2idx = label2idx
        self.img_feature_extractor = img_feature_extractor
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.data[index]['image_path'])
        img = Image.open(img_path).convert('RGB')

        if self.transforms:
            img = self.transforms(img)

        if self.img_feature_extractor:
            img = self.img_feature_extractor(images=img, return_tensors="pt")
            img = {k: v.to(self.device).squeeze(0) for k, v in img.items()}

        question = self.data[index]['question']
        if self.text_tokenizer:
            question = self.text_tokenizer(
                question,
                padding="max_length",
                max_length=20,
                truncation=True,
                return_tensors="pt"
            )
            question = {k: v.to(self.device).squeeze(0) for k, v in question.items()}

        label = self.data[index]['answer']
        label = torch.tensor(
            self.label2idx[label],
            dtype=torch.long
        ).to(self.device)

        sample = {
            'image': img,
            'question': question,
            'label': label
        }

        return sample
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.model = RobertaModel.from_pretrained("roberta-base")

    def forward(self, inputs):
        outputs = self.model(**inputs)
        return outputs.pooler_output
class VisualEncoder(nn.Module):
    def __init__(self):
        super(VisualEncoder, self).__init__()
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224")

    def forward(self, inputs):
        outputs = self.model(**inputs)
        return outputs.pooler_output
class Classifier(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        dropout_prob=0.2,
        n_classes=2
    ):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(768 * 2, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
class VQAModel(nn.Module):
    def __init__(
        self,
        visual_encoder,
        text_encoder,
        classifier
    ):
        super(VQAModel, self).__init__()
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.classifier = classifier

    def forward(self, image, answer):
        text_out = self.text_encoder(answer)
        image_out = self.visual_encoder(image)

        x = torch.cat((image_out, text_out), dim=1)
        x = self.classifier(x)

        return x

    def freeze(self, visual=True, textual=True, clas=False):
        if visual:
            for n, p in self.visual_encoder.named_parameters():
                p.requires_grad = False
        if textual:
            for n, p in self.text_encoder.named_parameters():
                p.requires_grad = False
        if clas:
            for n, p in self.classifier.named_parameters():
                p.requires_grad = False

def evaluate(model, dataloader, criterion):
    model.eval()
    correct = 0
    total = 0
    losses = []
    
    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            images = inputs['image']
            questions = inputs['question']
            labels = inputs['label']

            outputs = model(images, questions)
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
    epochs
):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        batch_train_losses = []

        model.train()
        for idx, inputs in enumerate(train_loader):
            images = inputs['image']
            questions = inputs['question']
            labels = inputs['label']

            optimizer.zero_grad()
            outputs = model(images, questions)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_train_losses.append(loss.item())

        train_loss = sum(batch_train_losses) / len(batch_train_losses)
        train_losses.append(train_loss)

        val_loss, val_acc = evaluate(
            model, val_loader,
            criterion
        )
        val_losses.append(val_loss)

        print(f'EPOCH {epoch + 1}:\tTrain loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}\tVal Acc: {val_acc:.4f}')

        scheduler.step()

    return train_losses, val_losses
               
if __name__ == '__main__':

    seed = 59
    set_seed(seed)
    # Load train data
    train_data = []
    train_set_path = '/home/always/VQA/Project-Visual-Question-Answering/data/vqa_coco_dataset/vaq2.0.TrainImages.txt'


        # Load train data
    train_data = []
    train_set_path = './vaq2.0.TrainImages.txt'

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
    val_set_path = './vaq2.0.DevImages.txt'

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
    test_set_path = './vaq2.0.TestImages.txt'

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

    classes = set([sample['answer'] for sample in train_data])
    label2idx = {
        cls_name: idx for idx, cls_name in enumerate(classes)
    }
    idx2label = {
        idx: cls_name for idx, cls_name in enumerate(classes)
    }

    data_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.CenterCrop(size=180),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.GaussianBlur(3),
])
    img_feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    text_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = VQADataset(
        train_data,
        label2idx=label2idx,
        img_feature_extractor=img_feature_extractor,
        text_tokenizer=text_tokenizer,
        device=device,
        transforms=data_transform
    )

    val_dataset = VQADataset(
        val_data,
        label2idx=label2idx,
        img_feature_extractor=img_feature_extractor,
        text_tokenizer=text_tokenizer,
        device=device
    )

    test_dataset = VQADataset(
        test_data,
        label2idx=label2idx,
        img_feature_extractor=img_feature_extractor,
        text_tokenizer=text_tokenizer,
        device=device
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
    hidden_size = 256
    dropout_prob = 0.2

    text_encoder = TextEncoder().to(device)
    visual_encoder = VisualEncoder().to(device)
    classifier = Classifier(
        hidden_size=hidden_size,
        dropout_prob=dropout_prob,
        n_classes=n_classes
    ).to(device)

    model = VQAModel(
        visual_encoder=visual_encoder,
        text_encoder=text_encoder,
        classifier=classifier
    ).to(device)

    model.freeze()

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
    epochs
)

    val_loss, val_acc = evaluate(
        model,
        val_loader,
        criterion
    )

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion
    )

    print('Evaluation on val/test dataset')
    print('Val accuracy:', val_acc)
    print('Test accuracy:', test_acc)




