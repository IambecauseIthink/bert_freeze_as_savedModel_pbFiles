export BERT_BASE_DIR=./GLUE/BERT_BASE_DIR/uncased_L-12_H-768_A-12
export FROZEN_MODEL_DIR=./GLUE/output/cola_output/pb_file
export TASK_DIR=./GLUE/glue_data/CoLA

python predict_bert_with_savedModel.py \
  --task_name=cola \
  --data_dir=$TASK_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --model_path=$FROZEN_MODEL_DIR \
  --max_seq_length=128 \