import numpy as np
import tensorflow as tf
import model


predicter = model.get_xlm_roberta('./xlm_roberta.h5')

input_file = "./inputs.txt"


if __name__ == '__main__':
    model.get_results(predicter, input_file)