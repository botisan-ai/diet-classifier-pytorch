start-featurizer:
	docker-compose up -d featurizer

infer:
	docker-compose run --rm diet-classifier python infer.py

train:
	docker-compose run --rm diet-classifier python train.py

clean-models:
	rm -rf lightning_logs

clean-featurizer:
	docker-compose down -v
