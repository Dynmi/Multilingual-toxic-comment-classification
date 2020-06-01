# Multilingual toxic comment classification
Identify toxicity in online conversations, where toxicity is defined as anything rude, disrespectful or otherwise likely to make someone leave a discussion.

## Required
- tensorflow2.0+
- numpy
- transformers
- tokenizers

## Model & Performance
- xlm roberta(default), which gets 0.93 score in JMTCC competition.
## Language support 
- English
- Italian
- Turkish
- Spanish

## How to use
1. clone this repository ``` git clone git@github.com:Dynmi/Multilingual-toxic-comment-classification.git ```
2. Download pre-trained weights here https://drive.google.com/file/d/1m1DyVXlk_6xImy7s9r4doLquUfsHXhhS/view?usp=sharing
3. put the comments in ```./inputs.txt```, one line for a single sentence
4. run ```predict.py```

## Reference
- https://huggingface.co/transformers/model_doc/xlmroberta.html
- https://www.kaggle.com/xhlulu/jigsaw-tpu-xlm-roberta
