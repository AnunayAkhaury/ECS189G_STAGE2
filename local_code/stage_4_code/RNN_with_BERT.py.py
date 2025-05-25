# RNN_with_BERT.py
"""
Enhanced RNN with BERT tokenizer and embeddings for better text generation
"""
import os
import torch
from torch.utils.data import Dataset, TensorDataset
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class BERTTokenizedDataset:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize_texts(self, texts):
        """Tokenize texts using BERT tokenizer"""
        all_tokens = []
        for text in texts:
            # Tokenize and convert to IDs
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            all_tokens.append(tokens.squeeze(0))

        return torch.stack(all_tokens)


def build_bert_lm_dataset(texts, tokenizer, seq_len, max_length=128):
    """Build language modeling dataset using BERT tokenizer"""
    examples = []

    for text in texts:
        # Tokenize the text
        tokens = tokenizer.encode(
            text,
            add_special_tokens=False,  # Don't add [CLS], [SEP] for LM
            truncation=True,
            max_length=max_length
        )

        # Create sliding window examples
        for i in range(len(tokens) - seq_len):
            input_seq = tokens[i:i + seq_len]
            target = tokens[i + seq_len]
            examples.append((input_seq, target))

    if not examples:
        print("No examples created! Text might be too short.")
        return None

    # Convert to tensors
    X = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
    y = torch.tensor([ex[1] for ex in examples], dtype=torch.long)

    return TensorDataset(X, y)


class BERTEnhancedRNN(nn.Module):
    """RNN Language Model with BERT embeddings"""

    def __init__(self, bert_model_name='bert-base-uncased',
                 rnn_units=256, num_layers=2, rnn_type='lstm',
                 dropout=0.3, freeze_bert=False):
        super().__init__()

        # Load BERT tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Freeze BERT if specified (saves memory, faster training)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Get BERT's embedding dimension
        bert_dim = self.bert.config.hidden_size  # 768 for bert-base
        vocab_size = self.bert.config.vocab_size

        # RNN layers
        rnn_kwargs = dict(
            input_size=bert_dim,
            hidden_size=rnn_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        if rnn_type == 'gru':
            self.rnn = nn.GRU(**rnn_kwargs)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(**rnn_kwargs)
        else:
            self.rnn = nn.LSTM(**rnn_kwargs)

        # Output layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_units, vocab_size)

    def forward(self, input_ids, attention_mask=None, hidden=None):
        # Get BERT embeddings
        with torch.no_grad() if hasattr(self, '_freeze_bert') else torch.enable_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = bert_outputs.last_hidden_state  # [batch, seq_len, 768]

        # Pass through RNN
        rnn_out, hidden = self.rnn(embeddings, hidden)

        # Apply dropout and output layer
        output = self.dropout(rnn_out)
        logits = self.fc(output)  # [batch, seq_len, vocab_size]

        return logits, hidden


def generate_text_bert(model, tokenizer, seed_text, gen_len=50,
                       seq_len=10, temperature=1.0, device=None):
    """Generate text using BERT-enhanced RNN"""
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Tokenize seed text
    seed_tokens = tokenizer.encode(seed_text, add_special_tokens=False)
    generated = seed_tokens.copy()

    # Generate tokens one by one
    for _ in range(gen_len):
        # Take last seq_len tokens as input
        input_seq = generated[-seq_len:]
        input_tensor = torch.tensor([input_seq], device=device)

        # Create attention mask (all 1s for simplicity)
        attention_mask = torch.ones_like(input_tensor)

        with torch.no_grad():
            logits, _ = model(input_tensor, attention_mask)

        # Get logits for last position
        next_token_logits = logits[0, -1, :] / temperature

        # Sample next token
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        # Add to generated sequence
        generated.append(next_token)

    # Decode back to text
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    return generated_text


# Easy replacement functions for your existing script
def build_vocab_bert(texts, tokenizer):
    """Compatibility function - BERT handles vocab internally"""
    # BERT tokenizer already has vocab
    vocab_size = tokenizer.vocab_size
    # Return dummy mappings for compatibility
    word2idx = {tokenizer.decode([i]): i for i in range(min(1000, vocab_size))}
    idx2word = {i: tokenizer.decode([i]) for i in range(min(1000, vocab_size))}
    return word2idx, idx2word


# Usage example
def create_bert_enhanced_model(bert_model='bert-base-uncased', freeze_bert=True):
    """Create a BERT-enhanced RNN model"""
    model = BERTEnhancedRNN(
        bert_model_name=bert_model,
        rnn_units=256,
        num_layers=2,
        rnn_type='lstm',
        dropout=0.3,
        freeze_bert=freeze_bert  # Set to True for faster training
    )
    return model


# Modified training loop (to be used in your script)
def train_bert_enhanced_model(model, train_loader, epochs=10, lr=0.001, device='cpu'):
    """Training function for BERT-enhanced model"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (input_ids, targets) in enumerate(train_loader):
            input_ids, targets = input_ids.to(device), targets.to(device)

            # Create attention mask
            attention_mask = (input_ids != 0).long()  # Assuming 0 is padding

            optimizer.zero_grad()

            # Forward pass
            logits, _ = model(input_ids, attention_mask)

            # Use last token's logits for prediction
            loss = criterion(logits[:, -1, :], targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return model