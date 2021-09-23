# Human Feedback for Query-Focused Abstractive Summarization

Our code is largely based on **[QAbsBert](https://github.com/rabimist/QABSBERT)**.
We omit information on data pre-processing and refer to (QAbsBert repo).
We have set up a [polybox](https://polybox.ethz.ch/index.php/s/hoHdmWHQVs3KBym) folder with the necassary files. The already pre-processed CNN/Dailymail for QAbsBert can be found there to download. 
To use the CNN/Dailymail place the .pt files in QABSBERT/bert_data or any other folder of your preference.

Take a look at our [project report](Human_Feedback_for_Query-Focused_Abstractive_Summarization.pdf) and [presentation](hf-qfas-presentation.pdf)!

## Dependencies

- Python 3.7.4
- PyTorch 1.9.0
- pytorch_transformers tensorboardX multiprocess pyrouge
- comet_ml, transformers, sklearn
- Further dependencies might be needed
- Evaluation requires a working ROUGE setup

## Reward Model
The code for our reward model consists of:
- rewardmodel_train.py
- rewardmodel_create_data.py
  - is used to precompute the embeddings from the raw OpenAI data
- rewardmodel/
    - open_ai_data/    
        - contains the raw human feedback data from OpenAI as well as the precomputed embeddings
    - config.py
        - lists all flags that can be used to configure the training, e.g. batch size, learning rate, comet_ml logging etc.
    - datasets.py
        - processing of the raw OpenAI data, dataloaders
    - model.py
        - contains the multi-layer perceptron reward model
    - trainer.py
        - setup of training/evaluation
    - utils.py
        - logging, embedding procedure

To train the rewardmodel go to the `QABSBERT/src` folder and run:
<pre><code>python rewardmodel_train.py</code></pre>

If you want to use all data set the `--trainsplit all` flag.
The trained model will be saved in the `QABSBERT/src/logs` folder.

## QAbsBert

### Training

For normal training of QAbsBert we refer to the project repository on [Github](https://github.com/rabimist/QABSBERT).

To run with a reward model:
Download the trained reward model and pre-trained QAbsBert from [polybox](https://polybox.ethz.ch/index.php/s/hoHdmWHQVs3KBym) or use your own trained model(s).
Go to the `QABSBERT/src` folder and run the following command to train QAbsBert with our reward model:
<pre><code>python train.py  -task abs -mode train -bert_data_path BERT_DATA_PATH/cnndm -dec_dropout 0.2  -model_path ../models -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 170000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0  -log_file ../logs/abs_bert_cnndm -train_from PATH_TO_PRETRAINED_QABSBERT -reward_model_ckpt PATH_TO_REWARDMODEL</code></pre>

Replace `PATH_TO_REWARDMODEL, PATH_TO_PRETRAINED_QABSBERT, BERT_DATA_PATH` with the correspoinding paths.

First run: For the first time, you should use -visible_gpus -1, so the code can download the BERT model. After downloading, you should kill the process and rerun the code with -visible_gpus 0 to run the training on 1 GPU.

### Evaluation
These instructions work for QAbsBert with and without our reward model.
Download the trained QAbsBert model from [polybox](https://polybox.ethz.ch/index.php/s/hoHdmWHQVs3KBym) or use your own trained model.

Go to the `QABSBERT/src` folder and run:

<pre><code>python train.py -task abs -mode test -batch_size 3000 -test_batch_size 500 -bert_data_path BERT_DATA_PATH/cnndm -log_file ../logs/val_abs_bert_cnndm -model_path ../models -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_cnndm -test_from PATH_TO_QABSBERT
</code></pre>

Replace `PATH_TO_QABSBERT, BERT_DATA_PATH` with the correspoinding paths.

**This step requires a working ROUGE setup**

## Running `preprocessReddit.py`
This script is adapted from `preprocessNewsroom.py` from [QAbsBert](https://github.com/rabimist/QABSBERT/tree/main/src). Read QAbsBert's README for more information.
### Download QAbsBert
Download QAbsBert from here. If you work with multiple datasets, download it again for each dataset.
### Download Reddit dataset
Download the dataset from [here](https://zenodo.org/record/1043504#.Wzt7PbhXryo), unzip it, and put it in QAbsBert's `raw_stories` folder.
Since the file is so big, do a test run using only the first couple of lines of the file.
### Stanford CoreNPL
Download Stanford CoreNLP version `3.9.2` from [here](https://stanfordnlp.github.io/CoreNLP/history.html).
Then add the following command to your bash_profile (with your path):
`export CLASSPATH=/path/to/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar`
Alternatively, uncomment and adjust the corresponding lines in `preprocessReddit.py`
### Run script
<pre><code>python preprocessReddit.py</code></pre>


## Leonhard good to knows

We run almost all our experiments on Leonhard. Here are some useful commands to setup the environment:

- To set up the environment use the following commands:
  <pre><code>module load python_gpu/3.7.4
  module load eth_proxy
  source $HOME/.local/bin/virtualenvwrapper.sh
  mkvirtualenv "env"</code></pre>
  Assuming you have virtualenvwrapper installed

- Most of the commands only work on GPU so precede every command with:

  <pre><code>bsub -n 4 -W 4:00 -R "rusage[mem=1024, ngpus_excl_p=1]" python ...</code></pre>

  Change it according to your needs.

- If any program is not be able to download BERT, run the program on CPU.
