import torch.nn as nn
import torch

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads):
        super().__init__()
        
        self.encoder_embedding = nn.Embedding(input_dim, hidden_dim)
        self.decoder_embedding = nn.Embedding(output_dim, hidden_dim)
        
        self.encoder_pos_embedding = nn.Embedding(1000, hidden_dim)
        self.decoder_pos_embedding = nn.Embedding(1000, hidden_dim)
        
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, encoder_input, decoder_input):
        encoder_embedded = self.encoder_embedding(encoder_input)
        decoder_embedded = self.decoder_embedding(decoder_input)
        
        encoder_positions = torch.arange(encoder_input.size(1), device=encoder_input.device).unsqueeze(0)
        decoder_positions = torch.arange(decoder_input.size(1), device=decoder_input.device).unsqueeze(0)
        
        encoder_embedded = encoder_embedded + self.encoder_pos_embedding(encoder_positions)
        decoder_embedded = decoder_embedded + self.decoder_pos_embedding(decoder_positions)
        
        for layer in self.encoder_layers:
            encoder_embedded = layer(encoder_embedded)
            
        encoder_output = encoder_embedded
        
        for layer in self.decoder_layers:
            decoder_embedded = layer(decoder_embedded, encoder_output)
        output = self.fc(decoder_embedded)
        return output
    

