import logging
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import string
import re

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    logger.info("Starting preprocessing.")

    input_data_path = os.path.join("/opt/ml/processing/input", "hate-speech-dataset.csv")

    try:
        os.makedirs("/opt/ml/processing/train")
        os.makedirs("/opt/ml/processing/validation")
    except:
        pass

    logger.info("Reading input data")

    # read csv
    df = pd.read_csv(input_data_path)

    new_tweets = []
    punct = "[!#$%&\'()*+,-./:';<=>?^_`{|}~]"

    hate_speech_list = []
    tweet_list = []

    def edit_text(txt):
        txt = re.sub(punct, '', txt)
        txt = re.sub(' RT', '. ', txt)
        txt = txt.lower()
        return txt

    for tweet in df['tweet']:
        txt = edit_text(tweet)
        tweet_list.append(txt)

    for hs in df['hate_speech']:
        if hs > 3:
            x = 3
        else:
            x = 3
        hate_speech_list.append(x)
    
    x_train, x_valid, y_train, y_valid = train_test_split(tweet_list, 
                                                          hate_speech_list, 
                                                          test_size=0.3, 
                                                          random_state=42)
    
    train_data = pd.DataFrame({'text':x_train, 'label':y_train})
    valid_data = pd.DataFrame({'text':x_valid, 'label':y_valid})
    
    train_data.to_csv("/opt/ml/processing/train/train.csv", header=False, index=False)
    validation_data.to_csv(
        "/opt/ml/processing/validation/validation.csv", header=False, index=False
    )
