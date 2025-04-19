import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, hidden, encoder_outputs):
        # hidden shape: (batch_size, hidden_dim)
        # encoder_outputs shape: (batch_size, seq_len, hidden_dim)
        
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_dim)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # (batch_size, seq_len)
        return torch.softmax(attention, dim=1)

class CaptioningModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, dropout=0.2):
        super().__init__()
        self.image_proj = nn.Linear(embedding_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_embeds=None, captions=None, valid_len=None, hidden=None):
        # Image feature processing
        if hidden is None:
            # Initial hidden state from image features
            h0 = self.image_proj(img_embeds).unsqueeze(0)  # (1, batch_size, hidden_dim)
            c0 = torch.zeros_like(h0)
            encoder_outputs = h0.permute(1, 0, 2)  # (batch_size, 1, hidden_dim)
        else:
            h0, c0 = hidden
            encoder_outputs = h0.permute(1, 0, 2)  # (batch_size, 1, hidden_dim)

        # Caption embedding
        if captions is not None:
            embedded = self.embedding(captions)  # (batch_size, seq_len, embedding_dim)
            
            # Attention weights
            attn_weights = self.attention(h0.squeeze(0), encoder_outputs)  # (batch_size, seq_len)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, hidden_dim)
            
            # Combine embeddings with context
            lstm_input = torch.cat([embedded, context.repeat(1, embedded.size(1), 1)], dim=2)
            
            # Handle packed sequences
            if valid_len is not None:
                lstm_input = pack_padded_sequence(lstm_input, valid_len.cpu(), 
                                                 batch_first=True, enforce_sorted=False)
        else:
            # Inference mode
            lstm_input = torch.cat([
                self.embedding(captions), 
                encoder_outputs.repeat(1, captions.size(1), 1)
            ], dim=2)

        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(lstm_input, (h0, c0))
        
        # Unpack if packed
        if valid_len is not None:
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            
        # Final output
        outputs = self.fc(self.dropout(lstm_out))
        return outputs, (hn, cn)
