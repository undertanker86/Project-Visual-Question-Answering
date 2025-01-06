# Project-Visual-Question-Answering
- `git clone https://github.com/undertanker86/Project-Visual-Question-Answering.git`
- Download data: https://drive.google.com/file/d/1kc6XNqHZJg27KeBuoAoYj70_1rT92191/view.
- Create Folder name: data and move dataset to this folder.
## 1. Visual Question Answering using CNN, LSTM, and FC
- `cd VQA_CNN+LSTM+FC` 
- `pip3 install -r requirements_cnn_lstm.txt`
- Adjust path of data in code (train, validation, test).
- `python main.py`
## 2. Visual Question Answering using ViT, RoBERTA, and FC
- `cd VQA_ViT+RoBERTa+FC` 
- `pip3 install -r requirements_vit_roberta.txt`
- Adjust path of data in code (train, validation, test).
- `python main.py`
## 3. Visual Question Answering using LLaVA
- `cd VQA_LLaVA_Inference` 
- `pip3 install -r requirements_llava.txt`
- Adjust path of data in code (test).
- `python main.py`
## Result:
![result](https://i.imgur.com/7g1VyIn.png)