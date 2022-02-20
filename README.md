# Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations

this paper present a simple and effective scheme for dependency parsing which is based on bidirectional-LSTMs (BiLSTMs).
Each sentence token is associated with a BiLSTM vector representing the token in its sentential context, and feature vectors are constructed by concatenating a few BiLSTM vectors.
The BiLSTM is trained jointly with the parser objective, resulting in very effective feature extractors for parsing.
