# Nowruz-at-SemEval-2022-Task-7
This repository contains the code used for participating in [SemEval-2022 Task 7](https://competitions.codalab.org/competitions/35210), our model ueses two ordinal regression heads with a pre-trained transformer as a backbone to tackle Task 7 of the SemEval 2022 which is "Identifying Plausible Clarifications of Implicit and Underspecified Phrases". our model officialy took the 5th and 7th place in sub-task A and B respectively.

# Abstract of Our Paper
This paper outlines the system using which team Nowruz participated in SemEval 2022 Task 7 “Identifying Plausible Clarifications of Implicit and Underspecified Phrases” for both subtasks A and B. Using a pre-trained transformer as a backbone, the model targeted the task of multi-task classification and ranking in the context of finding the best fillers for a cloze task related to instructional texts on the website Wikihow. 

The system employed a combination of two ordinal regression components to tackle this task in a multi-task learning scenario. According to the official leaderboard of the shared task, this system was ranked 5th in the ranking and 7th in the classification subtasks out of 21 participating teams. With additional experiments, the models have since been further optimised.

![](https://raw.githubusercontent.com/mohammadmahdinoori/Nowruz-at-SemEval-2022-Task-7/main/Figures/Figure.png)

# Results

### Best Results on Dev Set
| Model  | Best Acc  | Best Rank |
| :------------ |:-----:| :-----:|
| BERT       | 59.12% | 0.6341 |
| RoBERTa    | 60.20% | 0.6928 |
| DeBERTa-V3 | 64.12% | 0.7411 |
| T5         | 62.24% | 0.6949 |

### Best Results on Leaderboard
| Model  | Best Acc  | Best Rank |
| :------------ |:-----:| :-----:|
| RoBERTa    | 61.00% | 0.6700 |
| DeBERTa-V1 | 61.00% | 0.6900 |
| T5         | 62.40% | 0.7070 |

Check out all of the results in this [link](https://competitions.codalab.org/competitions/35210#results) (Click on Evaluation)

# Usage
In this section, we will explain how to use our code to reproduce our results as well as how to run experiments on your own datasets.

### Requirements

```bash
pip install transformers datasets sentencepiece coral_pytorch
```

Note that, the data_loader.py file should be next to Nowruz_SemEval.py since in the Nowruz_SemEval.py, a direct import is used.

### Setup
```python
from Nowruz_SemEval import *
import transformers as ts
```

### Loading Datasets
```python
trainDataset = loadDataset("Data/Train_Dataset.tsv",
                           labelPath="Data/Train_Labels.tsv", 
                           scoresPath="Data/Train_Scores.tsv")

valDataset = loadDataset("Data/Val_Dataset.tsv",
                         labelPath="Data/Val_Labels.tsv", 
                         scoresPath="Data/Val_Scores.tsv")

testDataset = loadDataset("Data/Test_Dataset.tsv")
```
`loadDataset` method is used for creating a Huggingface Dataset from tsv files provided in the shared task. it has one positional parameter and two optional parameters. <br/><br/>
`dataPath` is the first parameter and it is the path of the dataset <br/>
`labelPath` is the path of labels file for the dataset (if available) <br/>
`scoresPath` is the path of scores file for the dataset (if available) <br/>

### Initializing Tokenizer and DataCollator
```python
tokenizer = ts.AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
data_collator = ts.DataCollatorWithPadding(tokenizer=tokenizer , return_tensors="pt")
```
Since our code is based on Huggingface library you should use a pre-trained tokenizer from Huggingface or train your own tokenizer

### Preprocessing Datasets
```python
tokenizedTrainDataset = preprocessDataset(trainDataset , tokenizer)
tokenizedValDataset = preprocessDataset(valDataset , tokenizer)
tokenizedTestDataset = preprocessDataset(testDataset , tokenizer)
```
### Initializing Model
In order to initialize your model you have to select a pre-trained transformer as a backbone, to do so you have two options. First, you can use the pre-trained transformer path from Huggingface model zoo. Second, you can load pre-trained transformer yourself and then pass the model object to the `model_init` method.

#### Using Pre-Trained Transformer Path
```python
model = model_init(encoderPath="microsoft/deberta-v3-base",
                   dimKey="hidden_size",
                   mode="both",
                   use_coral=True, 
                   use_cls=True, 
                   supportPooledRepresentation=False,
                   freezeEmbedding=True, 
                   num_labels=3, 
                   num_ranks=5, 
                   lambda_c=0.5, 
                   lambda_r=0.5, 
                   dropout_rate=0.2,)
```

#### Loading Pre-Trained Transformer Yourself
```python
customEncoder = ts.T5EncoderModel.from_pretrained("t5-base")
customDim = customEncoder.config.to_dict()["d_model"]

#Freezing The Embedding Manually
customEncoder.shared.requires_grad = False
customEncoder.encoder.embed_tokens.requires_grad = False

model = model_init(customEncoder=customEncoder,
                   customDim=customDim,
                   mode="both", 
                   use_coral=True, 
                   use_cls=False, 
                   supportPooledRepresentation=False,
                   freezeEmbedding=False,
                   num_labels=3, 
                   num_ranks=5, 
                   lambda_c=0.5, 
                   lambda_r=0.5, 
                   dropout_rate=0.2,)
```

#### Parameters
- `encoderPath`: path of the pre-trained transformer which should be taken from the Huggingface model zoo.
- `dimKey`: the key of the hidden dimension size of the selected transformer which can be found in the config of the model. (to find this value just load the pre-trained model yourself and then print the value of `model.config`, However it is mostly `hidden_size` or `d_model`)
- `customEncoder`: the pre-trained transformer model object
- `customDim`: this is an int value which is the hidden dimension size of the pre-trained transformer
- `mode`: this property controls the training loss of the model. if the value is `both` the model is trained using a combined loss in a multi-task learning scenario (as defined in the paper). if the value is `classification` the model is trained using only the classification loss which is obtained from the labels. if the value is `regression` the model is trained using only the regression loss which is obtained from the scores.
- `use_coral `: if true the classification head would be a coral dense layer and the classification loss would be coral loss. (default is True)
- `use_cls`: this should be true if the model inserts a special token (like `[CLS]`) at the begining of each sentence otherwise it should be false. in BERT-Like models such as BERT, RoBERTa, DeBERTa, ELECTRA, and etc. this should be set to true but in models like GPT and T5 it should be set to false.
- `supportPooledRepresentation`: some models like BERT and RoBERTa use a dense layer after the last Transformer layer and pass the output of the `[CLS]` token to that dense layer followed by an activation function mostly `Tanh`. if the model has this option you should see a dense layer mostly called `Pooler` in the state_dict of the model. if the model has `Pooler` this property of the model_init function should be set to true otherwise it should be false.
- `freezeEmbedding`: set it to true if you want to freeze the embedding layer of the pre-trained transformer (default is true). note that if your transformer (e.g. T5) embedding layer has a name different from `embeddings` this property would not work and you should explicitly freeze the embedding layer before passing the transformer to `model_init` function.
- `num_labels`: number of plausibility labels which is `3` in the shared task dataset.
- `num_ranks`: number of different ranks which depends on the boundry used for converting continuos scores into discrete socres which is `5` if the boundry is `1` (ranks would be 1 , 2 , 3 , 4 , 5)
- `lambda_c`: the classification loss coeficiente used in the combined loss (this is only used if the mode of the model is set to `both`)
- `lambda_r`: the regression loss coeficiente used in the combined loss (this is only used if the mode of the model is set to `both`)
- `dropout_rate`: rate of the last dropout layer before passing the final output to the classification and regression heads.

Note that you should use one of the `encoderPath` and `customEncoder` and not both. it is also true for the `dimKey` and `customDim`
Note that `num_labels` and `num_ranks` should be set to `3` and `5` respectively while using our preprocessings for this task.
