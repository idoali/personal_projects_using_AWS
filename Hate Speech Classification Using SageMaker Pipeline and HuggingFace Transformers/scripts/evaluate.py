import json
import logging
import pathlib
import pickle
import tarfile
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    args, _ = parser.parse_known_args()
    
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall()

    logger.debug("Loading transformers model.")
    model = AutoModelForSequenceClassification.from_pretrained('/opt/ml/processing/input/model')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    logger.debug("Loading test input data.")
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)
    
    def predict(text):
        encode_plus_tokens = tokenizer.encode_plus(
            text, pad_to_max_length=True, max_length=args.max_seq_length, truncation=True, return_tensors="tf")
        input_ids = encode_plus_tokens["input_ids"]
        input_mask = encode_plus_tokens["attention_mask"]
        outputs = model.predict(x=(input_ids, input_mask))
        prediction = [{"label": item.argmax(), "score": item.max().item()} for item in outputs]
        return prediction[0]["label"]

    logger.debug("Reading test data.")
    y_test = df['label'].to_numpy()
    X_test = df['text'].values

    logger.info("Performing predictions against test data.")
    predictions = predict(X_test)
    
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    fpr, tpr, _ = roc_curve(y_test, prediction_probabilities)

    logger.debug("Accuracy: {}".format(accuracy))
    logger.debug("Precision: {}".format(precision))
    logger.debug("Recall: {}".format(recall))
    logger.debug("Confusion matrix: {}".format(conf_matrix))

    # Available metrics to add to model: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
            "precision": {"value": precision, "standard_deviation": "NaN"},
            "recall": {"value": recall, "standard_deviation": "NaN"},
            "confusion_matrix": {
                "0": {"0": int(conf_matrix[0][0]), "1": int(conf_matrix[0][1])},
                "1": {"0": int(conf_matrix[1][0]), "1": int(conf_matrix[1][1])},
            },
            "receiver_operating_characteristic_curve": {
                "false_positive_rates": list(fpr),
                "true_positive_rates": list(tpr),
            },
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))