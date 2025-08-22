all: all_build all_view_dim_reduction

all_build:
	python embedding_build.py --model_name all-MiniLM-L6-v2
	python embedding_build.py --model_name roberta-base-nli-stsb-mean-tokens

all_view_dim_reduction:
	python embedding_view_dim_reduction.py --downsample_fraction 0.1 --no-ui_show_blocking --model_name all-MiniLM-L6-v2 --fit_type UMAP
	python embedding_view_dim_reduction.py --downsample_fraction 0.1 --no-ui_show_blocking --model_name all-MiniLM-L6-v2 --fit_type t-SNE
	python embedding_view_dim_reduction.py --downsample_fraction 0.1 --no-ui_show_blocking --model_name roberta-base-nli-stsb-mean-tokens --fit_type UMAP
	python embedding_view_dim_reduction.py --downsample_fraction 0.1 --no-ui_show_blocking --model_name roberta-base-nli-stsb-mean-tokens --fit_type t-SNE

all_classify:
	python embedding_classify.py --model_name all-MiniLM-L6-v2
	python embedding_classify.py --model_name roberta-base-nli-stsb-mean-tokens