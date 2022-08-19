#!/bin/sh
python -m src.run \
	--problem_name mimic-iii_cl_50 \
	--checkpoint_dir /scratch/gobi2/wren/icd/laat/checkpoints \
	--max_seq_length 4000 \
	--n_epoch "1,1,1,1,50" \
	--patience 6 \
	--lr_scheduler_patience 2 \
	--batch_size 8 \
	--optimiser adamw \
	--lr 0.0005 \
	--dropout 0.3 \
	--main_metric micro_f1 \
	--save_results_on_train \
	--embedding_mode word2vec \
	--embedding_file data/embeddings/word2vec_sg0_100.model \
	--joint_mode hicu \
	--d_a 256 \
	--metric_level -1 \
	--loss ASL \
	--asl_config "1,0,0.03" \
	RNN \
	--rnn_mode LSTM \
	--n_layers 1 \
	--bidirectional 1 \
	--hidden_size 256