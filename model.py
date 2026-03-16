import torch.nn as nn

class IrisSmartNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=3, num_layers=2, dropout_rate=0.05):
        super(IrisSmartNN, self).__init__()

        # Input Layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.bn_input    = nn.BatchNorm1d(hidden_dim)
        
        # Hidden Layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )
        
        # Output Layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # ১. ইনপুট লেয়ার প্রসেসিং
        x = self.input_layer(x)
        x = self.bn_input(x)
        x = self.relu(x)
        x = self.dropout(x)
      
        # ২. হিডেন লেয়ার লুপ
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)
            x = self.batch_norms[i](x)
            x = self.relu(x)
            x = self.dropout(x)
            
        # ৩. আউটপুট লেয়ার (আউটপুটে সাধারণত অ্যাক্টিভেশন দেওয়া হয় না)
        x = self.output_layer(x)
        return x
