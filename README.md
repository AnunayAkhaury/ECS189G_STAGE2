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


\section*{Stage 4: RNN Models}

\subsection*{Model code location}
All RNN implementations live under \texttt{local\_code/stage\_4\_code/}:
\begin{itemize}
  \item \texttt{RNN.py} 
    \begin{itemize}
      \item \texttt{RNNClassifier} – sequence classification model  
      \item \texttt{RNNGenerator} – sequence generation model  
      \item \texttt{load\_data}, \texttt{setup\_glove\_embeddings}, \texttt{TextDataset} – data utilities
    \end{itemize}
\end{itemize}

\subsection*{Run classification training}
\begin{verbatim}
python script/stage_4_code/script_rnn_train.py
\end{verbatim}
This script will:
\begin{enumerate}
  \item Load train/test splits from \texttt{data/stage\_4\_data/text\_classification/}  
  \item Build vocabulary and (optionally) GloVe embeddings  
  \item Train \texttt{RNNClassifier}, evaluate accuracy/loss each epoch  
  \item Save checkpoint (\texttt{rnn\_model\_glove.pth}), history (\texttt{history\_glove.json}), and learning curves (\texttt{learning\_curves\_glove.png})
\end{enumerate}

\subsection*{Run generation}
\begin{verbatim}
python script/stage_4_code/script_rnn_generate.py
\end{verbatim}
This script will:
\begin{enumerate}
  \item Load the trained checkpoint (\texttt{rnn\_model\_glove.pth})  
  \item Given a text prompt, generate new sequences via \texttt{RNNGenerator}  
  \item Save outputs under \texttt{results/}
\end{enumerate}

\subsection*{To do}
\begin{itemize}
  \item Perform hyperparameter sweeps (e.g.\ learning rate, dropout, RNN type, bidirectionality)  
  \item Integrate an attention mechanism for longer context retention  
  \item Enhance generation with temperature sampling and beam search  
  \item Add unit tests for preprocessing, model save/load, and inference  
  \item Automate experiment logging \& result comparison
\end{itemize}
