# Stage 3 CNN Models

A quick reference to where the model classes live and how to launch their training scripts.

---

## Model code location

All CNN implementations are under `local_code/stage_3_code/`:

- **`CNN_MNIST.py`** – digits (MNIST)  
- **`CNN_ORL.py`** – faces (ORL)  
- **`CNN_CIFAR10.py`** – objects (CIFAR-10)  

Supporting utilities (accuracy evaluation, data loading, result saving) live in the same folder.

---

## Run training

Each script lives under `script/stage_3_code/` and can be launched directly:

  ```bash
  python script/stage_3_code/script_mlp_train_MNIST.py
  python script/stage_3_code/script_mlp_train_ORL.py
  python script/stage_3_code/script_cnn_train_cifar.py


Each script will:

Load its dataset from data/stage_3_data/

Instantiate the corresponding CNN_*.py class

Call .run() to train and evaluate

Save loss/accuracy plots and any model checkpoints under result/

```

# Stage 4: RNN Models

## Model code location
All RNN implementations live under `local_code/stage_4_code/`:

- **RNN.py**  
  - `RNNClassifier` – sequence classification model  
  - `RNNGenerator` – sequence generation model  
  - `load_data`, `setup_glove_embeddings`, `TextDataset` – data‐loading & preprocessing utilities  

Run through the setup colabs (PREFERRED):

Classification: https://colab.research.google.com/drive/13iPD5WPOLOPV-Psz_btxsdaGGL-Ckfvn

Generation: https://colab.research.google.com/drive/1D-DmPxNfA0kA-umM4ukFiKmk9W4iqULG

## Run classification/generation training
```bash
python script/stage_4_code/script_rnn_train.py

python script/stage_4_code/script_rnn_generate.py

```
# Stage 5: GCN Models

A quick reference to where the model classes live and how to launch their training scripts.

---

## Model code location

All GCN implementations are under `local_code/stage_5_code/`:

- **`GCN.py`** – two-layer Graph Convolutional Network model  
- **`Dataset_Loader_Node_Classification.py`** – citation-graph data loader  

---

## Run training

Script lives under `script/stage_5_code/` and can be launched directly:

```bash
python script/stage_5_code/script_gcn_train.py --dataset cora
python script/stage_5_code/script_gcn_train.py --dataset citeseer
python script/stage_5_code/script_gcn_train.py --dataset pubmed
```

Additonal Arguments for training: [-h] --dataset {cora,citeseer,pubmed} [--data_dir DATA_DIR] [--hidden_dim HIDDEN_DIM] [--lr LR] [--weight_decay WEIGHT_DECAY] [--dropout DROPOUT]
                           [--epochs EPOCHS] [--seed SEED]
Each run will:

Load data from data/stage_5_data/<dataset>/

Add self-loops and normalize the adjacency matrix

Instantiate the GCN model and move data to device

Train with Adam (weight decay + dropout), printing train/val (and test for PubMed) metrics per epoch

Save loss & accuracy plots and checkpoints under results/



