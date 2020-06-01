'''
author@DynmiWang

'''

import tensorflow as tf
import numpy as np
import transformers
from tokenizers import Tokenizer
max_len = 512

def regular_encode(texts, maxlen=512):
    tokenizer = transformers.AutoTokenizer.from_pretrained('jplu/tf-xlm-roberta-large')
    enc_di = tokenizer.batch_encode_plus(
        texts,
        return_attention_masks=False,
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    return np.array(enc_di['input_ids'])

def get_xlm_roberta(pre_weights = None):
    input_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_layer")
    transformer = transformers.TFAutoModel.from_pretrained('jplu/tf-xlm-roberta-large')
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:,0,:]
    out = tf.keras.layers.Dense(1, activation='sigmoid')(cls_token)
    model = tf.keras.Model(inputs=input_word_ids, outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    if pre_weights != None:
        model.load_weights(pre_weights)
    return model

def get_results(model, input_path, show_result=1, out_path=None, percent_format=0):
    in_wds = []
    results = []
    with open(input_path,"r") as f:
        line = f.readline()
        while line:
            in_wds.append(line)
            line = f.readline()
        results = model.predict( regular_encode(in_wds) )[0]

    if out_path!=None:
        with open(out_path,"a+") as f:
            for prob in results:
                if percent_format:
                    prob = str( round((float(prob[0]))*100,4) )+"%"
                else:
                    prob = str(float(prob) > 0.5)
                f.write(prob)

    if show_result:
        for prob in results:
            if percent_format:
                prob = str( round((float(prob[0]))*100,4) )+"%"
            else:
                prob = str(float(prob) > 0.5)
            print(prob)

    return results