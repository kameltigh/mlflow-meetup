curl --location 'http://127.0.0.1:5455/invocations' \
--header 'Content-Type: application/json' \
--data '{
        "dataframe_split": {
            "columns": ["input"],
            "data": ["Databricks, Inc. is an American enterprise software company founded by the creators of Apache Spark"]
        }
    }'