export BERT_BASE_DIR=./GLUE/BERT_BASE_DIR/uncased_L-12_H-768_A-12
export GLUE_DIR=./GLUE/glue_data
export OUTPUT_DIR=./GLUE/output/cola_output/

python run_classifier.py \
    --task_name=cola \
    --do_train=true \
    --do_eval=true \
    --data_dir=$GLUE_DIR/CoLA \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \                              --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=$OUTPUT_DIR

mv $OUTPUT_DIR/pb_file/[0-9]*[0-9]/* $OUTPUT_DIR/pb_file/
rm -r $OUTPUT_DIR/pb_file/[0-9]*[0-9]/
