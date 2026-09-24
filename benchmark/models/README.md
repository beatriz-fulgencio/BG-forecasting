# Forecasting models

Implemented architectures are RNN, LSTM, GRU, and a batch-first Transformer
encoder. The recurrent models form the original four-horizon benchmark. The
Transformer is a separately tuned 30-minute replication using 12 historical
five-minute samples, sinusoidal or learned positions, last-token or mean
pooling, and no causal mask because all encoder tokens are observed history.
