# Submission for mining frik competition

- setup base directory in main methods of preprocess.py and model.py files
	- base directory should contain ccdm_test.tsv, ccdm_sample.tsv, ccdm_medium.tsv and ccdm_large.tsv
- run preprocess.py (with python 2.7)
	- preprocess will create: ccdm_test-preprocessed.tsv and ccdm_all-preprocessed.tsv files in base directory
- run model.py
	- model will do crossvalidation, and predict the leaderboard set file in base directory:
		- prediction_xb_sparse.tsv
