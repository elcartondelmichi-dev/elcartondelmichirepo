import os
import sqlite3
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from deck_graph_builder import DeckGraphBuilder
from edh_gnn_model import EDHPowerGNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "prueba.db")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "edh_gnn_model.pt")
DECKS_DIR = os.path.join(BASE_DIR, "real_decks")


# -------------------------------------------------------------------------
# 1. LECTOR DE ARCHIVOS DE MAZOS REALES
# -------------------------------------------------------------------------
def cargar_mazo_desde_archivo(filepath):
  """Lee un archivo .txt con la lista de cartas de un mazo y respeta las cantidades."""
  card_names = []
  with open(filepath, 'r', encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith('#'):
        continue

      parts = line.split()
      count = 1
      first_token = parts[0].lower().replace('x', '')

      # Si el primer elemento es un número (ej. "32 Swamp" o "1x Sol Ring")
      if first_token.isdigit():
        count = int(first_token)
        card_name = ' '.join(parts[1:])
      else:
        card_name = line

      # Agregar la carta N veces según la cantidad indicada
      for _ in range(count):
        card_names.append(card_name)

  return card_names


# -------------------------------------------------------------------------
# 2. DATASET Y PIPELINE DE ENTRENAMIENTO
# -------------------------------------------------------------------------
class EDHRealDataset(Dataset):
    def __init__(self, db_path, decks_dir):
        self.builder = DeckGraphBuilder(db_path=db_path)
        self.data_list = []

        print(f"🔄 Cargando dataset de MAZOS REALES desde: {decks_dir}...")

        # Mapeo de carpetas a etiquetas de Bracket (0-indexed)
        bracket_folders = {
            "bracket_1": 0,
            "bracket_2": 1,
            "bracket_3": 2,
            "bracket_4": 3,
            "bracket_5": 4
        }

        total_decks = 0
        for folder_name, bracket_label in bracket_folders.items():
            folder_path = os.path.join(decks_dir, folder_name)
            
            if not os.path.exists(folder_path):
                print(f"⚠️ Advertencia: La carpeta '{folder_name}' no existe en '{decks_dir}'. Saltando...")
                continue

            files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
            print(f"   • Bracket {bracket_label + 1} ({folder_name}): Cargando {len(files)} mazos...")

            for file_name in files:
                file_path = os.path.join(folder_path, file_name)
                card_names = cargar_mazo_desde_archivo(file_path)

                if len(card_names) < 80:
                    print(f"⚠️ Mazo corto omitido ({len(card_names)} cartas): {file_name}")
                    continue

                # Construir el grafo utilizando la topología real del mazo
                x, edge_index = self.builder.build_graph_from_decklist(card_names)

                data_obj = Data(
                    x=x,
                    edge_index=edge_index,
                    y=torch.tensor([bracket_label], dtype=torch.long)
                )
                self.data_list.append(data_obj)
                total_decks += 1

        print(f"✅ Total de mazos reales procesados y convertidos a grafos: {total_decks}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"⚙️ Entrenando en el dispositivo: {device}")

    # Dataset basado en mazos reales
    dataset = EDHRealDataset(DB_PATH, DECKS_DIR)
    
    if len(dataset) == 0:
        print("❌ Error: No se encontraron mazos en 'real_decks/'. Agrega archivos .txt antes de entrenar.")
        return

    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Cambiar in_channels=23 por in_channels=24
    model = EDHPowerGNN(in_channels=28, hidden_channels=64, num_classes=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = 40
    print("\n🚀 Iniciando entrenamiento sobre Mazos Reales...")

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        correct = 0
        total_samples = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
            total_samples += batch.num_graphs

        acc = correct / total_samples
        avg_loss = total_loss / total_samples

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc*100:.2f}%")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ Modelo entrenado con datos reales guardado exitosamente en: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()