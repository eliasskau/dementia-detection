# ── Primary workflow: Docker (recommended) ───────────────────────────────
# PITT_DIR and RESULTS_DIR default to local paths; override as needed:
#   make docker-train PITT_DIR=/external/Pitt

IMAGE       = dementia-detection
PITT_DIR   ?= $(PWD)/Pitt
RESULTS_DIR ?= $(PWD)/results

# Local dev fallback (requires conda env dementia-detection active)
PYTHON = conda run -n dementia-detection python

.PHONY: preprocess features train search export evaluate pipeline clean \
        docker-build docker-test docker-train docker-evaluate docker-pipeline

# ── Docker (official) ─────────────────────────────────────────────────────
docker-build:
	docker build -t $(IMAGE) .

docker-test:
	docker run --rm \
		-v $(PITT_DIR):/app/Pitt \
		-v $(RESULTS_DIR):/app/results \
		$(IMAGE)

docker-train:
	docker run --rm \
		-v $(PITT_DIR):/app/Pitt \
		-v $(RESULTS_DIR):/app/results \
		$(IMAGE) conda run -n dementia-detection python pipeline.py train --task cookie
	docker run --rm \
		-v $(PITT_DIR):/app/Pitt \
		-v $(RESULTS_DIR):/app/results \
		$(IMAGE) conda run -n dementia-detection python pipeline.py export

docker-evaluate:
	docker run --rm \
		-v $(PITT_DIR):/app/Pitt \
		-v $(RESULTS_DIR):/app/results \
		$(IMAGE) conda run -n dementia-detection python pipeline.py evaluate

docker-pipeline: docker-train docker-evaluate

# ── Local dev (conda) ─────────────────────────────────────────────────────
preprocess:
	$(PYTHON) pipeline.py preprocess

features:
	$(PYTHON) pipeline.py features

train:
	$(PYTHON) pipeline.py train --task cookie
	$(PYTHON) pipeline.py export

search:
	$(PYTHON) pipeline.py search

evaluate:
	$(PYTHON) pipeline.py evaluate

pipeline: preprocess features train evaluate

clean:
	rm -rf results/models/cookie/*.pkl results/figures/*.png
