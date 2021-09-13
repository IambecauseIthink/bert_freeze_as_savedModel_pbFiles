## BERT

The original project built by google (Tensorflow edition) is in [this link](https://github.com/google-research/bert). Information about bert and pre-trained bert models can be found and download in the link.

## Freeze checkpoint model to pb (SavedModel)

- `checkpoint` files record parameters of the model, you can use it to do inference, and it can be continuesly trained if you need.

- `pb` stands for  protocol buffer. In TensorFlow, the pb file contains the graph definition as well as the weights of the model. Thus, a `pb` file is all you need to be able to run a given trained model.

  ### Why SavedModel?

  SavedModel is a language-neutral, recoverable, hermetic serialization format. SavedModel enables higher-level systems and tools to produce, consume, and transform TensorFlow models.

  When recording as SavedModel format, there is a saved_model.pb file and a variables folder, containing data and index.

## How to train and do inference

> Preparation:
>
> 1. Create a new folder and open it.
> 2. Clone the project.
> 3. Download the pre-trained [bert model](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip) in `/GLUE/BERT_BASE_DIR` and unzip it.
> 4. [**opinonal**] Download MNLI dataset in `/GLUE/glue_data/MNLI`.(It is too large to commit to github, so you may download it by yourself.)

The dataset is stored in `/GLUE/glue_data`, and can be trained using `start_classifier.sh` file. (In Linux, after granted permission, using `./start_classifier.sh` in this path.) The default and example is CoLA dataset. Nine datasets related to google's paper are present in `/GLUE/glue_data`. 

After training, you can use `bert_predict_pb.sh` to predict the probability of each label of the test set, and the results can be seen in `/GLUE/output/cola_output/test_result_pb.txt`. (The default one, you can change the directory in `bert_predict_pb.sh`, named as  `TASK_DIR` in the third line.)

## Processors of the nine datasets

I revised codes of run_classifier.py to train different datasets of glue. 

> All processors list below has updated in `run_classifier.py`, and you can choose one `Start training code` of any datasets below to replace  `start_classifier.sh`, and run it (the default one is CoLA). 

## MRPC

### Processor

```python
class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples
```

### Start training code

```sh
export BERT_BASE_DIR=./GLUE/BERT_BASE_DIR/uncased_L-12_H-768_A-12
export GLUE_DIR=./GLUE/glue_data
export OUTPUT_DIR=./GLUE/output/mrpc_output/

python run_classifier.py \
    --task_name=mrpc \
    --do_train=true \
    --do_eval=true \
    --data_dir=$GLUE_DIR/MRPC \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \                              	--max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=$OUTPUT_DIR

mv $OUTPUT_DIR/pb_file/[0-9]*[0-9]/* $OUTPUT_DIR/pb_file/
rm -r $OUTPUT_DIR/pb_file/[0-9]*[0-9]/
```

## CoLA

### Processor

```python
class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # Only the test set has a header
      if set_type == "test" and i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        label = "0"
      else:
        text_a = tokenization.convert_to_unicode(line[3])
        label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples
```

### Start training code

```sh
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
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \                              	--max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=$OUTPUT_DIR

mv $OUTPUT_DIR/pb_file/[0-9]*[0-9]/* $OUTPUT_DIR/pb_file/
rm -r $OUTPUT_DIR/pb_file/[0-9]*[0-9]/
```

## MNLI

### Processor

```python
class MnliProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
        "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[8])
      text_b = tokenization.convert_to_unicode(line[9])
      if set_type == "test":
        label = "contradiction"
      else:
        label = tokenization.convert_to_unicode(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples
```

### Start training code

```sh
export BERT_BASE_DIR=./GLUE/BERT_BASE_DIR/uncased_L-12_H-768_A-12
export GLUE_DIR=./GLUE/glue_data
export OUTPUT_DIR=./GLUE/output/mnli_output/

python run_classifier.py \
    --task_name=mnli \
    --do_train=true \
    --do_eval=true \
    --data_dir=$GLUE_DIR/MNLI \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \                              	--max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=$OUTPUT_DIR

mv $OUTPUT_DIR/pb_file/[0-9]*[0-9]/* $OUTPUT_DIR/pb_file/
rm -r $OUTPUT_DIR/pb_file/[0-9]*[0-9]/
```

## QNLI

### Processor

```python
class QnliProcessor(DataProcessor):
  """Processor for the QNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), 
        "dev_matched")
  
  # add
  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_test_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), 
        "test_matched")

  def get_labels(self):
    """See base class."""
    return ["entailment", "not_entailment"]

  # Train & Dev
  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, line[0])
      text_a = line[1]
      text_b = line[2]
      label = line[-1]
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  # Test
  def _create_test_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, line[0])
      text_a = line[1]
      text_b = line[2]
      label = "entailment" # padding
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples
```

### Start training code

```sh
export BERT_BASE_DIR=./GLUE/BERT_BASE_DIR/uncased_L-12_H-768_A-12
export GLUE_DIR=./GLUE/glue_data
export OUTPUT_DIR=./GLUE/output/qnli_output/

python run_classifier.py \
    --task_name=qnli \
    --do_train=true \
    --do_eval=true \
    --data_dir=$GLUE_DIR/QNLI \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \                              	--max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=$OUTPUT_DIR

mv $OUTPUT_DIR/pb_file/[0-9]*[0-9]/* $OUTPUT_DIR/pb_file/
rm -r $OUTPUT_DIR/pb_file/[0-9]*[0-9]/
```

## QQP

### Processor

```python
class QqpProcessor(DataProcessor):
  """Processor for the QQP data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the traininmg and dev sets."""
    examples = []
    try:
      for (i, line) in enumerate(lines):
        if i == 0:
          continue
        guid = "%s-%s" % (set_type, i)

        text_a = tokenization.convert_to_unicode(line[3])
        text_b = tokenization.convert_to_unicode(line[4])
        if set_type == "test":
          label = "0"
        else:
          label = tokenization.convert_to_unicode(line[-1])
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    except:
      print(guid)
    return examples
```

### Start training code

```sh
export BERT_BASE_DIR=./GLUE/BERT_BASE_DIR/uncased_L-12_H-768_A-12
export GLUE_DIR=./GLUE/glue_data
export OUTPUT_DIR=./GLUE/output/qqp_output/

python run_classifier.py \
    --task_name=qqp \
    --do_train=true \
    --do_eval=true \
    --data_dir=$GLUE_DIR/QQP \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \                              	--max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=$OUTPUT_DIR

mv $OUTPUT_DIR/pb_file/[0-9]*[0-9]/* $OUTPUT_DIR/pb_file/
rm -r $OUTPUT_DIR/pb_file/[0-9]*[0-9]/
```

## RTE

### Processor

```python
class RteProcessor(DataProcessor):
  """Processor for the RTE data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["entailment", "not_entailment"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the traininmg and dev sets."""
    examples = []
    try:
      for (i, line) in enumerate(lines):
        if i == 0:
          continue
        guid = "%s-%s" % (set_type, i)

        text_a = tokenization.convert_to_unicode(line[1])
        text_b = tokenization.convert_to_unicode(line[2])
        if set_type == "test":
          label = "not_entailment"
        else:
          label = tokenization.convert_to_unicode(line[-1])
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    except:
      print(guid)
    return examples
```

### Start training code

```sh
export BERT_BASE_DIR=./GLUE/BERT_BASE_DIR/uncased_L-12_H-768_A-12
export GLUE_DIR=./GLUE/glue_data
export OUTPUT_DIR=./GLUE/output/rte_output/

python run_classifier.py \
    --task_name=rte \
    --do_train=true \
    --do_eval=true \
    --data_dir=$GLUE_DIR/RTE \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \                              	--max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=$OUTPUT_DIR

mv $OUTPUT_DIR/pb_file/[0-9]*[0-9]/* $OUTPUT_DIR/pb_file/
rm -r $OUTPUT_DIR/pb_file/[0-9]*[0-9]/
```

## SST-2

### Processor

```python
class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)

            #test
            if set_type == "test":
              text_a = tokenization.convert_to_unicode(line[1])
              label = "0"
            #train & dev
            else:
              text_a = tokenization.convert_to_unicode(line[0])
              label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
```

### Start training code

```sh
export BERT_BASE_DIR=./GLUE/BERT_BASE_DIR/uncased_L-12_H-768_A-12
export GLUE_DIR=./GLUE/glue_data
export OUTPUT_DIR=./GLUE/output/sst2_output/

python run_classifier.py \
    --task_name=sst2 \
    --do_train=true \
    --do_eval=true \
    --data_dir=$GLUE_DIR/SST-2 \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \                              	--max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=$OUTPUT_DIR

mv $OUTPUT_DIR/pb_file/[0-9]*[0-9]/* $OUTPUT_DIR/pb_file/
rm -r $OUTPUT_DIR/pb_file/[0-9]*[0-9]/
```

## STS-B

> This dataset may be a regression problem with score from 0~5 in float. I divide them to 5 labels: 
>
> 0~1 -> "0"
>
> 1~2 -> "1"
>
> 2~3 -> "2"
>
> 3~4 -> "3"
>
> 4~5 -> "4"
>
> And make it a classify problem. However the results is not good now.

### Processor

```python
class StsbProcessor(DataProcessor):
  """Processor for the STS-B data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1", "2", "3", "4"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)

      text_a = tokenization.convert_to_unicode(line[7])
      text_b = tokenization.convert_to_unicode(line[8])
      if set_type == "test":
        label = "0"
      else:
        classify_label = "0"
        if float(line[-1]) > 1.0 and float(line[-1]) <= 2.0:
          classify_label = "1"
        elif float(line[-1]) > 2.0 and float(line[-1]) <= 3.0:
          classify_label = "2"
        elif float(line[-1]) > 3.0 and float(line[-1]) <= 4.0:
          classify_label = "3"
        elif float(line[-1]) > 4.0:
          classify_label = "4"
        label = classify_label
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples
```

### Start training code

```sh
export BERT_BASE_DIR=./GLUE/BERT_BASE_DIR/uncased_L-12_H-768_A-12
export GLUE_DIR=./GLUE/glue_data
export OUTPUT_DIR=./GLUE/output/stsb_output/

python run_classifier.py \
    --task_name=stsb \
    --do_train=true \
    --do_eval=true \
    --data_dir=$GLUE_DIR/STS-B \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \                              	--max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=$OUTPUT_DIR

mv $OUTPUT_DIR/pb_file/[0-9]*[0-9]/* $OUTPUT_DIR/pb_file/
rm -r $OUTPUT_DIR/pb_file/[0-9]*[0-9]/
```

## WNLI

### Processor

```python
class WnliProcessor(DataProcessor):
  """Processor for the WNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the traininmg and dev sets."""
    examples = []
    try:
      for (i, line) in enumerate(lines):
        if i == 0:
          continue
        guid = "%s-%s" % (set_type, i)

        text_a = tokenization.convert_to_unicode(line[1])
        text_b = tokenization.convert_to_unicode(line[2])
        if set_type == "test":
          label = "0"
        else:
          label = tokenization.convert_to_unicode(line[-1])
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    except:
      print(guid)
    return examples
```

### Start training code

```sh
export BERT_BASE_DIR=./GLUE/BERT_BASE_DIR/uncased_L-12_H-768_A-12
export GLUE_DIR=./GLUE/glue_data
export OUTPUT_DIR=./GLUE/output/wnli_output/

python run_classifier.py \
    --task_name=wnli \
    --do_train=true \
    --do_eval=true \
    --data_dir=$GLUE_DIR/WNLI \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \                              	--max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=$OUTPUT_DIR

mv $OUTPUT_DIR/pb_file/[0-9]*[0-9]/* $OUTPUT_DIR/pb_file/
rm -r $OUTPUT_DIR/pb_file/[0-9]*[0-9]/
```



