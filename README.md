# CSC 396 Final Project: Poem and Song Lyrics Sentiment Analysis

## Model info

### Labels Used
```python
labels = ['sad', 'joy', 'love', 'anger', 'fear', 'surprise']
```
Where
```python
0 = sad
1 = joy
2 = love
3 = anger
4 = fear
5 = surprise
```

### Training Structure
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

## Dataset Citations
### Kaggle Mood-Based English Songs Dataset
https://www.kaggle.com/datasets/nisharajayakody/mood-based-english-songs-dataset?resource=download

### PERC Dataset
Ponnarassery-, Sreeja (2017), “Poem Emotion Recognition Corpus (PERC)”, Mendeley Data, V1, doi: 10.17632/n9vbc8g9cx.1

### dair-ai Dataset
@inproceedings{saravia-etal-2018-carer,
    title = "{CARER}: Contextualized Affect Representations for Emotion Recognition",
    author = "Saravia, Elvis  and
      Liu, Hsien-Chi Toby  and
      Huang, Yen-Hao  and
      Wu, Junlin  and
      Chen, Yi-Shin",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D18-1404",
    doi = "10.18653/v1/D18-1404",
    pages = "3687--3697",
    abstract = "Emotions are expressed in nuanced ways, which varies by collective or individual experiences, knowledge, and beliefs. Therefore, to understand emotion, as conveyed through text, a robust mechanism capable of capturing and modeling different linguistic nuances and phenomena is needed. We propose a semi-supervised, graph-based algorithm to produce rich structural descriptors which serve as the building blocks for constructing contextualized affect representations from text. The pattern-based representations are further enriched with word embeddings and evaluated through several emotion recognition tasks. Our experimental results demonstrate that the proposed method outperforms state-of-the-art techniques on emotion recognition tasks.",
}

### mteb Dataset
@misc{sheng2020investigating,
  archiveprefix = {arXiv},
  author = {Emily Sheng and David Uthus},
  eprint = {2011.02686},
  primaryclass = {cs.CL},
  title = {Investigating Societal Biases in a Poetry Composition System},
  year = {2020},
}


@article{enevoldsen2025mmtebmassivemultilingualtext,
  title={MMTEB: Massive Multilingual Text Embedding Benchmark},
  author={Kenneth Enevoldsen and Isaac Chung and Imene Kerboua and Márton Kardos and Ashwin Mathur and David Stap and Jay Gala and Wissam Siblini and Dominik Krzemiński and Genta Indra Winata and Saba Sturua and Saiteja Utpala and Mathieu Ciancone and Marion Schaeffer and Gabriel Sequeira and Diganta Misra and Shreeya Dhakal and Jonathan Rystrøm and Roman Solomatin and Ömer Çağatan and Akash Kundu and Martin Bernstorff and Shitao Xiao and Akshita Sukhlecha and Bhavish Pahwa and Rafał Poświata and Kranthi Kiran GV and Shawon Ashraf and Daniel Auras and Björn Plüster and Jan Philipp Harries and Loïc Magne and Isabelle Mohr and Mariya Hendriksen and Dawei Zhu and Hippolyte Gisserot-Boukhlef and Tom Aarsen and Jan Kostkan and Konrad Wojtasik and Taemin Lee and Marek Šuppa and Crystina Zhang and Roberta Rocca and Mohammed Hamdy and Andrianos Michail and John Yang and Manuel Faysse and Aleksei Vatolin and Nandan Thakur and Manan Dey and Dipam Vasani and Pranjal Chitale and Simone Tedeschi and Nguyen Tai and Artem Snegirev and Michael Günther and Mengzhou Xia and Weijia Shi and Xing Han Lù and Jordan Clive and Gayatri Krishnakumar and Anna Maksimova and Silvan Wehrli and Maria Tikhonova and Henil Panchal and Aleksandr Abramov and Malte Ostendorff and Zheng Liu and Simon Clematide and Lester James Miranda and Alena Fenogenova and Guangyu Song and Ruqiya Bin Safi and Wen-Ding Li and Alessia Borghini and Federico Cassano and Hongjin Su and Jimmy Lin and Howard Yen and Lasse Hansen and Sara Hooker and Chenghao Xiao and Vaibhav Adlakha and Orion Weller and Siva Reddy and Niklas Muennighoff},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2502.13595},
  year={2025},
  url={https://arxiv.org/abs/2502.13595},
  doi = {10.48550/arXiv.2502.13595},
}

@article{muennighoff2022mteb,
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Lo{\"\i}c and Reimers, Nils},
  title = {MTEB: Massive Text Embedding Benchmark},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2210.07316},
  year = {2022}
  url = {https://arxiv.org/abs/2210.07316},
  doi = {10.48550/ARXIV.2210.07316},
}
