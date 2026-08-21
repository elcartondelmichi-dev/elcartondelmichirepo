import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from deck_graph_builder import DeckGraphBuilder
from edh_gnn_model import EDHPowerGNN
from deck_parser import parse_moxfield_deck

# --- LISTA MOXFIELD DE SHORIKAI (BRACKET 3 - HIGH POWER) ---
shorikai_moxfield = """
1 Heroic Intervention
1 Akroma's Memorial
1 Assassin's Trophy
1 Beast Whisperer
1 Beast Within
1 Beseech the Mirror
1 Bloodline Bidding
1 Canopy Tactician
1 Casualties of War
1 Command Tower
1 Cover of Darkness
1 Craterhoof Behemoth
1 Deathcap Glade
1 Deathrite Shaman
1 Defense of the Heart
1 Diabolic Tutor
1 Dwynen, Gilt-Leaf Daen
1 Eclipsed Realms
1 Elderfang Venom
1 Elven Ambush
1 Elvish Archdruid
1 Elvish Champion
1 Elvish Guidance
1 Elvish Harbinger
1 Elvish Mystic
1 Elvish Promenade
1 Elvish Warmaster
1 Exotic Orchard
1 Ezuri, Renegade Leader
1 Farhaven Elf
1 Fog
1 Forest
1 Forest
1 Forest
1 Forest
1 Forest
1 Forest
1 Forest
1 Forest
2 Forest
1 Fyndhorn Elves
1 Galadhrim Ambush
1 Genesis Wave
1 Gilt-Leaf Palace
1 Gloom Ripper
1 Growing Rites of Itlimoc // Itlimoc, Cradle of the Sun
1 Harald, King of Skemfar
1 Heritage Druid
1 Immaculate Magistrate
1 Imperious Perfect
1 Invasion of Ikoria // Zilortha, Apex of Ikoria
1 Joraga Warcaller
1 Leaf-Crowned Visionary
1 Llanowar Elves
1 Llanowar Wastes
1 Lys Alana Huntmaster
1 Marwyn, the Nurturer
1 Morcant's Loyalist
1 Nissa Revane
1 Overwhelming Stampede
1 Pact of the Serpent
1 Path of Ancestry
1 Poison-Tip Archer
1 Priest of Titania
1 Quirion Ranger
1 Reclamation Sage
1 Ruthless Winnower
1 Shaman of the Pack
1 Skemfar Shadowsage
1 Sol Ring
1 Staff of Domination
1 Swamp
1 Swamp
1 Swamp
1 Swamp
1 Swamp
1 Swamp
1 Swamp
1 Swamp
1 Swamp
1 Swamp
1 Swamp
1 Swamp
1 Taunting Elf
1 Throne of the God-Pharaoh
1 Timberwatch Elf
1 Tyvar Kell
1 Tyvar the Bellicose
1 Tyvar, Jubilant Brawler
1 Virulent Emissary
1 Wellwisher
1 Wirewood Channeler
1 Wirewood Lodge
1 Wirewood Pride
1 Wirewood Symbiote
1 Wolverine Riders
1 Woodland Cemetery
1 Yavimaya, Cradle of Growth
1 Lathril, Blade of the Elves
"""

# --- EJEMPLO SINTÉTICO CASUAL / BATTLECRUISER (BRACKET 1) ---
casual_deck = [
    "Plains", "Island", "Forest", "Mountain", "Swamp",
    "Cultivate", "Rampant Growth", "Colossal Dreadmaw",
    "Serra Angel", "Grizzly Bears", "Hill Giant", "Llanowar Elves"
]

def entrenar_modelo():
    print("🧠 Parseando mazos y construyendo Dataset de Entrenamiento...\n")
    builder = DeckGraphBuilder()

    shorikai_cards = parse_moxfield_deck(shorikai_moxfield)
    
    # 1. Creamos los Grafos
    x_shorikai, edge_shorikai = builder.build_graph_from_decklist(shorikai_cards)
    x_casual, edge_casual = builder.build_graph_from_decklist(casual_deck)

    # 2. Asignamos Etiquetas Target (0-indexed):
    # Bracket 1 (Battlecruiser) -> Índice 0
    # Bracket 3 (High Power)    -> Índice 2
    dataset = [
        (x_casual, edge_casual, torch.tensor([0])),     # Bracket 1
        (x_shorikai, edge_shorikai, torch.tensor([2]))  # Bracket 3
    ]

    # 3. Inicializamos Modelo, Pérdida y Optimizador
    model = EDHPowerGNN(in_channels=19, hidden_channels=32, num_classes=4)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print("🚀 Iniciando Backpropagation / Entrenamiento de la GNN...\n")
    model.train()

    for epoch in range(1, 61):
        total_loss = 0
        for x, edge_index, target in dataset:
            optimizer.zero_grad()
            output = model(x, edge_index)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"   • Época {epoch:02d}/60 | Loss (Error de la red): {total_loss:.4f}")

    print("\n✅ Entrenamiento Completo. Evaluando Shorikai nuevamente...\n")
    
    model.eval()
    with torch.no_grad():
        out = model(x_shorikai, edge_shorikai)
        probs = F.softmax(out, dim=1).numpy()[0]

    brackets = ["Bracket 1 (Battlecruiser)", "Bracket 2 (Mid Power)", "Bracket 3 (High Power)", "Bracket 4 (cEDH)"]
    print("📊 --- PREDICCIÓN AJUSTADA PARA TU SHORIKAI CONTROL ---")
    for i, p in enumerate(probs):
        barra = "█" * int(p * 20)
        print(f"   • {brackets[i].ljust(28)}: {p*100:5.1f}% | {barra}")

if __name__ == "__main__":
    entrenar_modelo()