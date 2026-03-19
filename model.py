# Định nghĩa kiến trúc mô hình Transformer cải tiến - FIXED VERSION
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from config import *

class PositionEncoding(nn.Module):
    def __init__(self, d_model=D_MODEL, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROPOUT_RATE):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        if mask is not None:
            # Ensure mask has correct dimensions: (batch_size, num_heads, seq_len, seq_len)
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(context)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=D_MODEL, d_ff=2048, dropout=DROPOUT_RATE):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class DecoderLayer(nn.Module):
    def __init__(self, d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=2048, dropout=DROPOUT_RATE):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class FootballChatbot(L.LightningModule):
    def __init__(self, num_tokens, d_model=D_MODEL, max_len=MAX_LEN):
        super().__init__()
        self.save_hyperparameters()
        
        self.we = nn.Embedding(num_embeddings=num_tokens, embedding_dim=d_model)        
        self.pe = PositionEncoding(d_model=d_model, max_len=max_len)
        self.decoders = nn.ModuleList([
            DecoderLayer(d_model, NUM_HEADS) for _ in range(NUM_LAYERS)
        ])
        self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)  # <PAD> luôn là 0
        
    def create_causal_mask(self, seq_len):
        """Tạo causal mask để che phần tương lai - FIXED VERSION"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
        return mask.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)
        
    def forward(self, token_ids):
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
            
        batch_size, seq_len = token_ids.shape
        
        word_embeddings = self.we(token_ids)
        position_encoded = self.pe(word_embeddings)
        x = self.dropout(position_encoded)
        
        # Tạo mask với đúng kích thước
        causal_mask = self.create_causal_mask(seq_len)
        
        for decoder in self.decoders:
            x = decoder(x, causal_mask)
            
        output = self.fc_layer(x)
        return output
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=LEARNING_RATE, 
            betas=(0.9, 0.98), 
            eps=1e-9,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        input_tokens, labels = batch
        
        # Debug kích thước
        if self.global_step == 0:
            self.print(f"Input shape: {input_tokens.shape}")
            self.print(f"Labels shape: {labels.shape}")
        
        output = self.forward(input_tokens)
        
        # Reshape output và labels cho loss calculation
        output_flat = output.reshape(-1, output.size(-1))
        labels_flat = labels.reshape(-1)
        
        loss = self.loss(output_flat, labels_flat)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        # Log learning rate
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            self.log('learning_rate', scheduler.get_last_lr()[0], prog_bar=True)