# Multilingual toxic comment classification
Identify toxicity in online conversations, where toxicity is defined as anything rude, disrespectful or otherwise likely to make someone leave a discussion.

## Required
- tensorflow2.0+
- numpy
- transformers
- tokenizers

## Model for use
- xlm roberta(default), which gets 0.91 score in JMTCC competition.
## Language support 
- English
- Italian
- Franch
- Russian

## How to use
1. clone this repository ``` git clone git@github.com:Dynmi/Multilingual-toxic-comment-classification.git ```
2. Download pre-trained weights here
3. put the comments in ```./inputs.txt```, one line for a single sentence
4. run ```predict.py```
