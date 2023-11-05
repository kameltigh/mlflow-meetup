import fasttext
import re

if __name__ == "__main__":
    model = fasttext.load_model("models/28-10-2023_18-16.bin")
    prediction = model.predict(
        "Databricks, Inc. is an American enterprise software company founded by the creators of Apache Spark.[2] "
        "Databricks develops a web-based platform for working with Spark, that provides automated cluster management "
        "and IPython-style notebooks. The company develops Delta Lake, an open-source project to bring reliability to "
        "data lakes for machine learning and other data science use cases.[3]", k=1, threshold=.9)[0]
    if len(prediction) > 0:
        prediction = re.sub("__label__", "", prediction[0])
        prediction = re.sub("-", " ", prediction)
        print(prediction)
    else:
        print(None)
