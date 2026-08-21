import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

class EDHPowerGNN(nn.Module):
    def __init__(self, in_channels=23, hidden_channels=64, num_classes=5, dropout_rate=0.2):
        """
        GNN optimizada para clasificación de nivel de poder en Commander (EDH).
        Incluye conexiones residuales y LayerNorm para prevenir Oversmoothing.
        """
        super(EDHPowerGNN, self).__init__()
        
        # Proyección inicial para llevar las 23 características al espacio oculto
        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        
        # Capas de Convolución en Grafos
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # Normalización por Capa (mucho más estable en GNNs que BatchNorm1d)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.norm3 = nn.LayerNorm(hidden_channels)
        
        # Clasificador Multicapa (MLP)
        # hidden_channels * 2 por la combinación de Mean Pool + Max Pool
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, num_classes)
        
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 0. Proyección inicial del nodo
        h = F.relu(self.node_encoder(x))

        # 1. Bloque Conv 1 + Residual
        h_res = h
        h = self.conv1(h, edge_index)
        h = self.norm1(h)
        h = F.relu(h + h_res)  # Conexión residual
        h = self.dropout(h)

        # 2. Bloque Conv 2 + Residual
        h_res = h
        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        h = F.relu(h + h_res)  # Conexión residual
        h = self.dropout(h)

        # 3. Bloque Conv 3 + Residual
        h_res = h
        h = self.conv3(h, edge_index)
        h = self.norm3(h)
        h = F.relu(h + h_res)  # Conexión residual

        # 4. Pooling Global Doble (Deck-level Embedding)
        x_mean = global_mean_pool(h, batch)
        x_max = global_max_pool(h, batch)
        x_deck = torch.cat([x_mean, x_max], dim=1)  # [batch_size, hidden_channels * 2]

        # 5. MLP de Clasificación
        x_deck = F.relu(self.fc1(x_deck))
        x_deck = self.dropout(x_deck)
        out = self.fc2(x_deck)
        
        return out

# --- PRUEBA DE ESTRUCTURA Y FORMAS ---
if __name__ == "__main__":
    num_nodes = 100
    num_features = 28
    
    x_dummy = torch.randn(num_nodes, num_features)
    
    # Generamos un edge_index fully connected de 100 nodos (9900 aristas)
    nodes = torch.arange(num_nodes)
    grid_x, grid_y = torch.meshgrid(nodes, nodes, indexing='ij')
    sources, targets = grid_x.flatten(), grid_y.flatten()
    mask = sources != targets
    edge_index_dummy = torch.stack([sources[mask], targets[mask]], dim=0)
    
    model = EDHPowerGNN(in_channels=28, hidden_channels=64, num_classes=5)
    output = model(x_dummy, edge_index_dummy)
    
    print(f"✅ Formato de entrada: {x_dummy.shape}")
    print(f"✅ Formato de salida (Logits): {output.shape}")
    print(f"Predicción (Logits no normalizados):\n{output.detach().numpy()}")