# Stage 1: Finetune RoBERTa with general text and emotion

## Training Structure
Datasets with columns:
```python
Dataset({
    features: ['text', 'label'],
    num_rows: n
})
```
Where text is either individual lines of poems or song lyrics, and label is dtype int64

## How to Use
### 1) Import the base sentiment model
```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification

name = './roberta-base-sentiments'
tokenizer = RobertaTokenizer.from_pretrained(name)
model = RobertaForSequenceClassification.from_pretrained(name)
```
### 2) Fine tune towards either poems or songs
Example
```python
from transformers import TrainingArguments, Trainer

trainging_args = TrainingArguments( ... )
trainer = Trainer( ... )
trainer.Train()
```
### 3) Test on new songs or poems
```python
from transformers import Trainer

def tokenize(dataset):
    return tokenizer(dataset['text'], truncation=True, padding='max_length')
test1 = songs.map(
    tokenize, batched=True,
)
test1 = test1.cast_column("label", Value("int64"))

output = trainer.predict(dataset)
```