import torch
import torch.nn.functional as F
from models.gnn_model import LocalGNN


class FederatedClient:
    def __init__(self, client_id, data, input_dim, hidden_dim, output_dim, lr=0.01, device="cpu"):
        self.client_id = client_id
        self.data = data.to(device)
        self.device = device
        self.model = LocalGNN(input_dim, hidden_dim, output_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)

    def _get_training_mask(self):
        if hasattr(self.data, "train_mask"):
            mask = self.data.train_mask
            if mask is not None and int(mask.sum()) > 0:
                return mask
        return torch.ones(self.data.num_nodes, dtype=torch.bool, device=self.data.x.device)

    def train(self, epochs=1):
        self.model.train()
        mask = self._get_training_mask()
        last_loss = None
        for _ in range(epochs):
            self.optimizer.zero_grad()
            out = self.model(self.data)
            loss = F.cross_entropy(out[mask], self.data.y[mask])
            loss.backward()
            self.optimizer.step()
            last_loss = loss
        return float(last_loss.item())

    def get_weights(self):
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights):
        self.model.load_state_dict(weights)
