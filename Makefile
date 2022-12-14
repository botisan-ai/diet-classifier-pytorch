start-featurizer:
	docker-compose up -d featurizer

infer:
	docker-compose run --rm diet-classifier python infer.py

train:
	docker-compose run --rm diet-classifier python train.py

train-debug:
	docker-compose run -p 5678:5678 --rm diet-classifier python -m debugpy --listen 0.0.0.0:5678 train.py

clean-models:
	rm -rf lightning_logs

clean-featurizer:
	docker-compose down -v
