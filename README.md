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
