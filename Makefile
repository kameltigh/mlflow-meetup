start_mlflow_server:
	poetry run mlflow ui --port 9911

create-serving-venv:
	virtualenv .serving-venv --python=3.11

serve_production_model:
	MLFLOW_TRACKING_URI="http://127.0.0.1:9911" mlflow models serve --port 5455 -m "models:/gpt_detector/Production"

generate-dockerfile:
	MLFLOW_TRACKING_URI="http://127.0.0.1:9911" mlflow models generate-dockerfile -d docker -m "models:/gpt_detector/Production"

call_served_model:
	./call_model.sh