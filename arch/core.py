import torch
from torch import nn
from lib.hyper_parameters import hyper_parameters as HP


class Core(nn.Module):
    def __init__(self, device='cuda'):
        super(Core, self).__init__()
        self.fc = nn.Linear(HP.total_embedding_size, HP.embedding_size)

    def forward(self, x):
        return self.fc(x)


class Core_gru(nn.Module):
    def __init__(self, device='cuda'):
        super(Core_gru, self).__init__()
        self.gru = nn.GRU(input_size=HP.embedding_size, hidden_size=HP.hidden_size, num_layers=HP.lstm_layers,
                          batch_first=True)
        self.hidden_size = HP.hidden_size
        self.n_layers = HP.lstm_layers
        self.batch_size = HP.batch_size
        self.device = device

    def forward(self, embedded_scalar, embedded_entity, embedded_spatial, hidden_state=None):
        input_tensor = torch.cat([embedded_scalar, embedded_entity, embedded_spatial], dim=1)
        input_tensor = input_tensor.reshape(1, self.batch_size, HP.embedding_size)

        hidden_state, _ = self.gru(input_tensor, hidden_state)
        return hidden_state


#   def forward_gru(self, x, hidden):
#       out, hidden = self.gru(x, hidden)
#       return out, hidden

#   def init_hidden(self, batch_size = 1):
#       self.hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device),
#                      torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device))


class Core_lstm(nn.Module):
    def __init__(self, device='cuda', drop_prob=0):
        super(Core, self).__init__()
        self.lstm = nn.LSTM(input_size=HP.embedding_dim, hidden_size=HP.hidden_size, num_layers=HP.lstm_layers,
                            dropout=drop_prob,
                            batch_first=True)
        self.hidden = None
        self.hidden_size = HP.hidden_size
        self.n_layers = HP.lstm_layers
        self.batch_size = HP.batch_size
        self.seq_length = HP.seq_length
        self.device = device

        self.init_hidden()

    def forward(self, embedded_scalar, embedded_entity, embedded_spatial):
        batch_seq_size = embedded_scalar.shape[0]
        input_tensor = torch.cat([embedded_scalar, embedded_entity, embedded_spatial], dim=1)
        embedding_size = input_tensor.shape[-1]
        input_tensor = input_tensor.reshape(self.batch_size, self.seq_length, embedding_size)

        out, self.hidden = self.forward_lstm(input_tensor)
        out = out.reshape(self.batch_size * self.seq_length, self.hidden_size)
        return out

    def forward_lstm(self, x):
        out, hidden = self.lstm(x, self.hidden)
        return out, hidden

    def init_hidden(self, batch_size=1):
        self.hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device),
                       torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device))
