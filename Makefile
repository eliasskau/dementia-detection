PYTHON = conda run -n dementia-detection python

.PHONY: pipeline preprocess features train evaluate clean

pipeline: preprocess features train evaluate

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
	$(PYTHON) scripts/09_hyperparameter_search.py --model svm
	$(PYTHON) scripts/10_export_best_model.py

evaluate:
	$(PYTHON) scripts/11_evaluate_best_model.py

clean:
	rm -rf results/models/cookie/*.pkl results/figures/*.png
