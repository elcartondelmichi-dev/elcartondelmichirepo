import os
import torch
import torch.nn.functional as F

from deck_parser import parse_moxfield_deck
from deck_graph_builder import DeckGraphBuilder
from edh_gnn_model import EDHPowerGNN
from mapeo_nlp import extraer_hallazgos_28d,construir_reporte_llm


BRACKET_NAMES = [
    "Bracket 1: Exhibition (Ultra-Casual)",
    "Bracket 2: Core (Precons)",
    "Bracket 3: Upgraded (Mid-High Power)",
    "Bracket 4: Optimized (High Power)",
    "Bracket 5: cEDH (Competitive Metagame)"
]

# Nombres exactos de las 28 características del vector de nodos (CardFeatureExtractor 28D)
FEATURE_NAMES = [
    "CMC Normalizado", "Color Blanco (W)", "Color Azul (U)", "Color Negro (B)", "Color Rojo (R)", "Color Verde (G)",
    "Es Criatura", "Es Tierra", "Es Artefacto", "Es Encantamiento", "Es Instante", "Es Conjuro", "Es Planeswalker",
    "Ramp Tradicional", "Fast Mana / Rituals", "Motor de Robo (Draw)", "Fast Tutor (CMC <= 2)", "Slow Tutor (CMC >= 3)",
    "Removal Directo", "Counterspell", "Board Wipe", "Turnos Extra", "Mass Land Denial (MLD)", "Spell Copy / Storm",
    "Cheat Mana / Reanimate", "Burn / Drain / Ping", "Multiplicadores de Disparo", "Game Changer Flag"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "edh_gnn_model.pt")


def evaluar_mazo(deck_text: str, model_path: str = None, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    print("🧠 Parseando lista e identificando cartas...")
    raw_cards = parse_moxfield_deck(deck_text)
    
    # 1. CONSTRUCCIÓN DEL GRAFO DESDE LA BASE DE DATOS
    builder = DeckGraphBuilder()
    x_target, edge_target = builder.build_graph_from_decklist(raw_cards)
    
    if x_target.size(0) == 0:
        print("❌ Error: No se pudieron extraer nodos del mazo.")
        return

    # Mover tensores al dispositivo objetivo (CPU/GPU/MPS)
    x_target = x_target.to(device)
    edge_target = edge_target.to(device)

    # Contar Game Changers automáticamente (Dimensión 28 -> índice 27)
    game_changer_flags = x_target[:, 27]
    gc_count = int(torch.sum(game_changer_flags).item())

    # 2. CARGA DEL MODELO GNN (28 Canales de Entrada)
    model = EDHPowerGNN(in_channels=28, hidden_channels=64, num_classes=5).to(device)
    
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print(f"📦 Modelo cargado exitosamente desde {model_path}")
        except Exception as e:
            print(f"⚠️ Error al cargar el modelo: {e}. Se usarán pesos aleatorios.")
    else:
        print("⚠️ Evaluando con arquitectura sin entrenar (pesos aleatorios)...")

    # 3. CREAR TENSOR DE BATCH
    batch_vector = torch.zeros(x_target.size(0), dtype=torch.long, device=device)

    # 4. INFERENCIA PURA
    model.eval()
    with torch.no_grad():
        out = model(x_target, edge_target, batch_vector)
        
        # Temperatura 1.0 (Sin distorsión artificial de probabilidades)
        temperatura = 0.6  
        probs = F.softmax(out / temperatura, dim=1).cpu().numpy()[0]

    # 5. DIAGNÓSTICO Y EXPLICABILIDAD VECTORIAL DE LA GNN
    feature_activations = torch.sum(x_target, dim=0).cpu().numpy()
    top_indices = feature_activations.argsort()[::-1][:5]

    # 6. DESPLIEGUE DE RESULTADOS
    print("\n" + "="*65)
    print("        📊 EVALUADOR OFICIAL DE BRACKETS WOTC (GNN Vectorial)")
    print("="*65)
    print(f"🃏 Total de cartas procesadas : {len(raw_cards)}")
    print(f"⚠️ Game Changers Detectados   : {gc_count} (vía Feature 28)")
    print(f"🖥️ Dispositivo de Inferencia  : {device.upper()}")
    
    predicted_bracket = probs.argmax()
    print(f"\n🏆 BRACKET PREDICHO PRINCIPAL: {BRACKET_NAMES[predicted_bracket]}")
    
    print("\n📈 DISTRIBUCIÓN DE PROBABILIDAD POR BRACKET:")
    for i, p in enumerate(probs):
        barra = "█" * int(p * 25)
        print(f"   • {BRACKET_NAMES[i].ljust(42)}: {p*100:5.1f}% | {barra}")

    # Generar el reporte una sola vez fuera del bucle
    hallazgos = extraer_hallazgos_28d(x_target, raw_cards)
    reporte_ia = construir_reporte_llm(probs, hallazgos, nombre_mazo="Chatterfang Aristocrats")

    print("\n" + reporte_ia)

    print("\n🔍 ACTIVACIÓN ESTRUCTURAL EN LA GNN (Top Features en Grafo):")
    print("-" * 65)
    for idx in top_indices:
        nombre_feat = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"Feature #{idx}"
        activacion = feature_activations[idx]
        print(f"   • {nombre_feat.ljust(35)} -> Densidad acumulada: {activacion:.1f}")

    print("="*65 + "\n")


if __name__ == "__main__":
    mi_mazo = """
1 Arcane Signet
1 Atarka, World Render
1 Barkchannel Pathway
1 Beast Within
1 Blasphemous Act
1 Bloom Tender
1 Bloomvine Regent
1 Broodcaller Scourge
1 Chaos Warp
1 Cinder Glade
1 Command Tower
1 Counterspell
1 Cultivate
1 Dragon Tempest
1 Dragon's Hoard
1 Dragonlord Atarka
1 Dragonlord's Servant
1 Dragonmaster Outcast
1 Dragonspeaker Shaman
1 Drakuseth, Maw of Flames
1 Dreamroot Cascade
1 Elemental Bond
1 Encroaching Dragonstorm
1 Enduring Courage
1 Eshki Dragonclaw
1 Exotic Orchard
1 Farseek
1 Flooded Grove
1 Fog
5 Forest
1 Freed from the Real
1 Frontier Bivouac
1 Frostcliff Siege
1 Garruk's Uprising
1 Genesis Wave
1 Ghalta, Primal Hunger
1 Glorybringer
1 Goreclaw, Terror of Qal Sisma
1 Grafted Exoskeleton
1 Hammerhead Tyrant
1 Haven of the Spirit Dragon
1 Heroic Intervention
1 Hinterland Harbor
5 Island
1 Kodama's Reach
1 Lathliss, Dragon Queen
1 Leyline Tyrant
1 Marang River Regent
1 Miirym, Sentinel Wyrm
1 Mosswort Bridge
6 Mountain
1 Nogi, Draco-Zealot
1 Parapet Thrasher
1 Radagast of Rhosgobel
1 Rapid Hybridization
1 Redirect Lightning
1 Reliquary Tower
1 Rhythm of the Wild
1 Riverglide Pathway
1 Rockfall Vale
1 Rootbound Crag
1 Sakura-Tribe Elder
1 Sarkhan, Soul Aflame
1 Selvala, Heart of the Wilds
1 Sol Ring
1 Steam Vents
1 Stomping Ground
1 Stormbreath Dragon
1 Stormcarved Coast
1 Stormscale Scion
1 Sulfur Falls
1 Summon: Fenrir
1 Temple of the Dragon Queen
1 Temur Ascendancy
1 Temur Battlecrier
1 Territorial Hellkite
1 Terror of the Peaks
1 The Earth King
1 Thunderbreak Regent
1 Thundermane Dragon
1 Topiary Stomper
1 Twinflame Tyrant
1 Ureni of the Unwritten
1 Verix Bladewing
1 Vineglimmer Snarl
1 Worldly Tutor
1 Eshki, Temur's Roar
    """
    evaluar_mazo(mi_mazo, model_path=MODEL_PATH)