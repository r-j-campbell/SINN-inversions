class SINN_MLP(nn.Module):
	"""
	Multi-layer perceptron for spectropolarimetric inversion.

	Args:
	    input_dim (int): Dimensionality of the input vector (e.g., flattened Stokes profiles).
	    hidden_dim (int): Size of the hidden layers.
	    output_dim (int): Number of output parameters (e.g., 6 physical quantities).
	    num_layers (int): Total number of hidden layers (not including input/output layers).

	Architecture:
	    - Input layer followed by SiLU activation.
	    - (num_layers - 1) hidden layers, each with SiLU activation.
	    - Final output layer is linear, with no activation function.

	Notes:
	    - This model is intended for regression. The inputs and outputs should be scaled before training,
	      and inverse scaling should be applied after inference.
	    - No activation function is applied after the final layer to allow unbounded real-valued outputs.
	    - This version does not contain dropout.
	"""
	def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
		super().__init__()
		layers = []
		layers.append(nn.Linear(input_dim, hidden_dim))
		layers.append(nn.SiLU())

		for _ in range(num_layers - 1):
			layers.append(nn.Linear(hidden_dim, hidden_dim))
			layers.append(nn.SiLU())

		layers.append(nn.Linear(hidden_dim, output_dim))
		self.model = nn.Sequential(*layers)

	def forward(self, x):
		return self.model(x)


class SINN_Transformer_Sequence(nn.Module):
	"""
	Transformer-based model for stratified spectropolarimetric inversion.

	Args:
	    seq_len (int): Number of spectral input tokens (Stokes vectors).
	    input_dim (int): Dimensionality of each input token.
	    n_tau (int): Number of optical depth layers to infer.
	    hidden_dim (int): Dimension of the internal embedding space.
	    num_layers (int, optional): Number of layers in both encoder and decoder. Default is 4.
	    num_heads (int, optional): Number of attention heads in encoder and decoder. Default is 4.

	Architecture:
	    - Linear projection + learned positional encoding for input sequence.
	    - Transformer encoder processes the input sequence (e.g., Stokes profiles).
	    - Transformer decoder takes a learned optical depth query embedding and cross-attends to the encoder output.
	    - Final linear head maps decoder output to physical parameters.

	Outputs:
	    - Tensor of shape (batch_size, n_tau, 6), representing six physical parameters 
	      (e.g., temperature, magnetic field strength, LOS velocity, inclination, sin(2φ), cos(2φ)) as a function of optical depth.

	Notes:
	    - This model uses cross-attention to relate spectral input to stratified atmospheric output.
	    - The 6 output parameters are hardcoded but can be adapted if needed.
	    - Inputs and outputs should be scaled before training, and inverse-scaled after inference.
	"""
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

		self.output_head = nn.Linear(hidden_dim, 6) #output dim is hardcoded

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
