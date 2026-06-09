# ── Configuration ────────────────────────────────────────────────────────────
IMAGE       = dementia-detection
PITT_DIR   ?= $(PWD)/Pitt
RESULTS_DIR ?= $(PWD)/results

DOCKER_RUN = docker run --rm \
	-v $(PITT_DIR):/app/Pitt \
	-v $(RESULTS_DIR):/app/results \
	$(IMAGE)

.PHONY: build test train search export evaluate pipeline clean \
        docker-build docker-test docker-train docker-evaluate docker-pipeline

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t $(IMAGE) .

docker-test:
	$(DOCKER_RUN)

docker-train:
	$(DOCKER_RUN) python pipeline.py train --task cookie
	$(DOCKER_RUN) python pipeline.py export

docker-evaluate:
	$(DOCKER_RUN) python pipeline.py evaluate

docker-pipeline: docker-train docker-evaluate

# ── Local (activate your env first: conda activate dementia-detection) ────────
train:
	python pipeline.py train --task cookie
	python pipeline.py export

search:
	python pipeline.py search

evaluate:
	python pipeline.py evaluate

test:
	python -m pytest tests/ -v

pipeline: train evaluate

clean:
	rm -rf results/models/cookie/*.pkl results/figures/*.png

clean:
	rm -rf results/models/cookie/*.pkl results/figures/*.png
