import datetime
import os
import re
from collections import defaultdict
from pathlib import Path

import fasttext
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec
from mlflow.types.schema import Schema

import pandas as pd

model_id = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")
dataset_path = Path("data")
dataset_path.mkdir(exist_ok=True)
TRAIN_DATASET_PATH = str(dataset_path / "dataset.train")
EVAL_DATASET_PATH = str(dataset_path / "dataset.eval")
DATASET_SOURCE_URL = "https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro/resolve/main/GPT-wiki-intro.csv.zip"


class FastTextMLFModel(mlflow.pyfunc.PythonModel):
    PROB_THRESHOLD = 0.9

    def __init__(self, params):
        self.model = None
        self.params = params

    def load_context(self, context):
        self.model = fasttext.load_model(context.artifacts["model_path"])

    def clear_context(self):
        self.model = None

    def __predict_label(self, desc):
        try:
            prediction = self.model.predict(
                desc, threshold=self.PROB_THRESHOLD
            )[0]
            if len(prediction) > 0:
                prediction = re.sub("__label__", "", prediction[0])
                prediction = re.sub("-", " ", prediction)
            else:
                prediction = None

        except Exception as inst:
            prediction = f"ERROR:{str(inst)}"

        return prediction

    def train(self):
        self.model = fasttext.train_supervised(
            input=self.params["train_input"],
            lr=self.params.get("lr", 0.3),
            dim=self.params.get("dim", 100),
            ws=self.params.get("ws", 5),
            epoch=self.params.get("epoch", 5),
            minCount=self.params.get("minCount", 1),
            minCountLabel=self.params.get("minCountLabel", 1),
            minn=self.params.get("minn", 0),
            maxn=self.params.get("maxn", 0),
            neg=self.params.get("neg", 5),
            wordNgrams=self.params.get("wordNgrams", 5),
            loss=self.params.get("loss", "softmax"),
            bucket=self.params.get("bucket", 2000000),
            thread=self.params.get("thread", 4),
            lrUpdateRate=self.params.get("lrUpdateRate", 100),
            t=self.params.get("t", 0.0001),
            label=self.params.get("label", "__label__"),
            verbose=self.params.get("verbose", 2),
        )

        self.model.save_model(self.params["model_location"])

    def evaluate(self, eval_data_path: str):
        n, p, r = self.model.test(eval_data_path, k=1, threshold=self.PROB_THRESHOLD)
        return {
            "Sample_count": n,
            "Precision": p,
            "Recall": r,
        }

    def predict(self, context, input_data):
        tmp = input_data

        if "input" not in input_data.columns:
            tmp.columns = ["input"]

        result = tmp["input"].apply(lambda x: self.__predict_label(x))

        return result


def add_row_to_dict(row, dict_to_add):
    """
    Adding a row to a dictionary
    :param row: row to add
    :type row: string
    :param dict_to_add: dictionary to add to
    :type dict_to_add: dict
    """
    dict_to_add["wiki"].append(row["wiki_intro"])
    dict_to_add["generated"].append(row["generated_intro"])


def prepare_dataset(dataset_path: str, train_ratio: int = 0.8):
    """
    Preparing the dataset for training
    :param train_ratio: ratio of the dataset to use for training
    :type train_ratio: int
    """
    df = pd.read_csv(dataset_path)

    train_dict = defaultdict(list)
    eval_dict = defaultdict(list)

    for idx, row in df.iterrows():
        if len(train_dict["wiki"]) < train_ratio * len(df):
            add_row_to_dict(row, train_dict)
        else:
            add_row_to_dict(row, eval_dict)

    with open(TRAIN_DATASET_PATH, "w") as output_file:
        for category_name, category_lines in train_dict.items():
            for line in category_lines:
                output_file.write(f"__label__{category_name} {line}\n")
    with open(EVAL_DATASET_PATH, "w") as output_file:
        for category_name, category_lines in eval_dict.items():
            for line in category_lines:
                output_file.write(f"__label__{category_name} {line}\n")

    return df


if __name__ == "__main__":
    parameters = {
        "train_input": TRAIN_DATASET_PATH,
        "eval_input": EVAL_DATASET_PATH,
        "model_location": f"models/{model_id}.bin",
        "lr": 0.3,
        "dim": 30,
        "ws": 5,
        "epoch": 1,
        "minCount": 6,
        "minCountLabel": 1,
        "minn": 3,
        "maxn": 0,
        "neg": 5,
        "wordNgrams": 4,
        "loss": "softmax",
        "bucket": 4000000,
        "threads": 8,
        "lrUpdateRate": 100,
        "t": 0.0001,
        "label": "__label__",
        "verbose": 3,
        "prob_threshold": FastTextMLFModel.PROB_THRESHOLD,
    }

    Path(parameters["model_location"]).parent.mkdir(parents=True, exist_ok=True)

    os.environ["MLFLOW_GCS_DEFAULT_TIMEOUT"] = str(
        60 * 10
    )  # 10 minutes max upload time

    mlflow.set_tracking_uri("http://127.0.0.1:9911")
    mlflow.set_experiment("GPT_detector")
    with mlflow.start_run(run_name="fasttext-model") as run:
        df = prepare_dataset(dataset_path=DATASET_SOURCE_URL)

        train_dataset = mlflow.data.from_pandas(df, source=DATASET_SOURCE_URL)

        mlflow.log_input(train_dataset)

        # log parameters
        mlflow.log_params(parameters)

        # train model
        fasttextMLF = FastTextMLFModel(parameters)
        fasttextMLF.train()

        # evaluate model
        metrics = fasttextMLF.evaluate(EVAL_DATASET_PATH)
        mlflow.log_metrics(metrics)

        # log model
        input_schema = Schema([ColSpec("string", "input")])
        output_schema = Schema([ColSpec("string", "line_category")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        fasttextMLF.clear_context()

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=fasttextMLF,
            signature=signature,
            artifacts={"model_path": parameters["model_location"]},
        )

        mlflow.log_artifact(local_path=TRAIN_DATASET_PATH)
        mlflow.log_artifact(local_path=EVAL_DATASET_PATH)

        # return mlflow run
        run_id = run.info.run_id
