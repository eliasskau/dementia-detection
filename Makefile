# ── Primary workflow: Docker (recommended) ───────────────────────────────
# PITT_DIR and RESULTS_DIR default to local paths; override as needed:
#   make docker-train PITT_DIR=/external/Pitt

IMAGE       = dementia-detection
PITT_DIR   ?= $(PWD)/Pitt
RESULTS_DIR ?= $(PWD)/results

# Local dev fallback (requires conda env dementia-detection active)
PYTHON = conda run -n dementia-detection python

.PHONY: pipeline preprocess features train evaluate clean \
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
		$(IMAGE) conda run -n dementia-detection python scripts/08_train_models.py --task cookie
	docker run --rm \
		-v $(PITT_DIR):/app/Pitt \
		-v $(RESULTS_DIR):/app/results \
		$(IMAGE) conda run -n dementia-detection python scripts/10_export_best_model.py

docker-evaluate:
	docker run --rm \
		-v $(PITT_DIR):/app/Pitt \
		-v $(RESULTS_DIR):/app/results \
		$(IMAGE) conda run -n dementia-detection python scripts/11_evaluate_best_model.py

docker-pipeline: docker-train docker-evaluate

# ── Local dev (conda) ─────────────────────────────────────────────────────
preprocess:
	$(PYTHON) scripts/01_preprocess_transcripts.py
	$(PYTHON) scripts/02_extract_participant_audio.py

features:
	$(PYTHON) scripts/03_extract_linguistic_features.py
	$(PYTHON) scripts/04_extract_acoustic_features.py
	$(PYTHON) scripts/05_integrate_liwc.py
	$(PYTHON) scripts/06_add_response_length.py
	$(PYTHON) scripts/07_combine_features.py

train:
	$(PYTHON) scripts/08_train_models.py --task cookie
	$(PYTHON) scripts/10_export_best_model.py

evaluate:
	$(PYTHON) scripts/11_evaluate_best_model.py

pipeline: preprocess features train evaluate

clean:
	rm -rf results/models/cookie/*.pkl results/figures/*.png
