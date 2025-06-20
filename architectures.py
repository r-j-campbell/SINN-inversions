class SINN_MLP(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super().__init__()
		self.model = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.SiLU(),
			nn.Linear(hidden_dim, hidden_dim // 2),
			nn.LeakyReLU(),
			nn.Linear(hidden_dim // 2, output_dim)
		)

	def forward(self, x):
		return self.model(x)


class SINN_Transformer_Sequence(nn.Module):
	def __init__(self, seq_len, input_dim, n_tau, hidden_dim, num_layers=4, num_heads=4):
		super().__init__()
		self.hidden_dim = hidden_dim
		self.n_tau = n_tau
		self.seq_len = seq_len

		self.embedding = nn.Linear(input_dim, hidden_dim)
		self.positional_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))

		encoder_layer = nn.TransformerEncoderLayer(
			d_model=hidden_dim,
			nhead=num_heads,
			batch_first=True
		)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

		decoder_layer = nn.TransformerDecoderLayer(
			d_model=hidden_dim,
			nhead=num_heads,
			batch_first=True
		)
		self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

		self.query_content = nn.Parameter(torch.randn(n_tau, hidden_dim))     # learned content
		self.depth_pos_enc = nn.Parameter(torch.randn(n_tau, hidden_dim))     # learned positional encoding

		self.output_head = nn.Linear(hidden_dim, 6)

	def forward(self, x):
		# Encode the spectral sequence
		x = self.embedding(x) + self.positional_embedding  # (batch, seq_len, hidden_dim)
		memory = self.encoder(x)  # (batch, seq_len, hidden_dim)

		# Prepare query with content + depth encoding
		query = self.query_content + self.depth_pos_enc  # (n_tau, hidden_dim)
		query = query.unsqueeze(0).expand(x.size(0), -1, -1)  # (batch, n_tau, hidden_dim)

		# Cross-attend to encoder output
		out = self.decoder(tgt=query, memory=memory)  # (batch, n_tau, hidden_dim)

		return self.output_head(out)  # (batch, n_tau, 6)
