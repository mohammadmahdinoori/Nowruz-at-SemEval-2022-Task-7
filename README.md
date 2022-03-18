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
from nowruz_semeval import *
import transformers as ts
```
