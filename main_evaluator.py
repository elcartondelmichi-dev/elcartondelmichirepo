import os
import torch
import torch.nn.functional as F

from deck_parser import parse_moxfield_deck
from deck_graph_builder import DeckGraphBuilder
from edh_gnn_model import EDHPowerGNN
from mapeo_nlp import obtener_cartas_clave_por_feature, construir_reporte_humano


BRACKET_NAMES = [
    "Bracket 1: Exhibition (Ultra-Casual)",
    "Bracket 2: Core (Precons)",
    "Bracket 3: Upgraded (Mid-High Power)",
    "Bracket 4: Optimized (High Power)",
    "Bracket 5: cEDH (Competitive Metagame)"
]

# Nombres de las 23 características del vector de nodos que procesa el DeckGraphBuilder
FEATURE_NAMES = [
    "CMC / Curva de Maná", "Es Criatura", "Es Artefacto", "Es Encantamiento",
    "Es Conjuro/Instante", "Es Tierra", "Produce Maná", "Es Tutor/Búsqueda",
    "Es Board Wipe/Removal", "Es Motor de Robo", "Efecto Untap/Sinergia", 
    "Potencial de Combo", "Fast Mana", "Interacción Barata", "Sinergia Tribal", 
    "Contadores / Sinergias", "Poder/Resistencia", "Incoloro", "Blanco", 
    "Azul", "Negro", "Rojo", "Game Changer Flag"
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

    # Contar Game Changers automáticamente desde el vector (Dimensión 23 -> índice 22)
    game_changer_flags = x_target[:, 22]
    gc_count = int(torch.sum(game_changer_flags).item())

    # 2. CARGA DEL MODELO GNN
    model = EDHPowerGNN(in_channels=26, hidden_channels=64, num_classes=5).to(device)
    
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
        
        # Factor de temperatura: < 1.0 afila el pico de mayor confianza
        temperatura = 0.6  
        probs = F.softmax(out / temperatura, dim=1).cpu().numpy()[0]

    # 5. DIAGNÓSTICO Y EXPLICABILIDAD VECTORIAL DE LA GNN
    # Sumamos la activación de cada feature a lo largo de todos los nodos del grafo
    feature_activations = torch.sum(x_target, dim=0).cpu().numpy()
    
    # Obtenemos las 5 dimensiones con mayor peso en el grafo (excluyendo tipos básicos como Tierras si se desea)
    top_indices = feature_activations.argsort()[::-1][:5]

    # 6. DESPLIEGUE DE RESULTADOS
    print("\n" + "="*65)
    print("        📊 EVALUADOR OFICIAL DE BRACKETS WOTC (GNN Vectorial)")
    print("="*65)
    print(f"🃏 Total de cartas procesadas : {len(raw_cards)}")
    print(f"⚠️ Game Changers Detectados   : {gc_count} (vía Feature 23)")
    print(f"🖥️ Dispositivo de Inferencia  : {device.upper()}")
    
    predicted_bracket = probs.argmax()
    print(f"\n🏆 BRACKET PREDICHO PRINCIPAL: {BRACKET_NAMES[predicted_bracket]}")
    
    print("\n📈 DISTRIBUCIÓN DE PROBABILIDAD POR BRACKET:")
    for i, p in enumerate(probs):
        barra = "█" * int(p * 25)
        print(f"   • {BRACKET_NAMES[i].ljust(42)}: {p*100:5.1f}% | {barra}")

        hallazgos = obtener_cartas_clave_por_feature(x_target, raw_cards)
        reporte_texto = construir_reporte_humano(probs, hallazgos)

    print("\n" + reporte_texto)

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
1 Ayara, First of Locthwain
1 Bastion of Remembrance
1 Big Apple, 3 a.m.
1 Black Market
1 Black Market Connections
1 Blood Artist
1 Bontu's Monument
1 Burglar Rat
1 Castle Locthwain
1 Corrupted Conviction
1 Crypt Rats
1 Culling the Weak
1 Dark Confidant
1 Dark Prophecy
1 Dark Ritual
1 Darkness
1 Deadly Dispute
1 Diabolic Tutor
1 Diseased Vermin
1 Doom Blade
1 Emeritus of Woe
1 Exsanguinate
1 Force of Despair
1 Foul-Tongue Shriek
1 Gnat Miser
1 Go for the Throat
1 Golbez, Clad In Darkness
1 Gray Merchant of Asphodel
1 Gruesome Fate
1 Ink-Eyes, Servant of Oni
1 Karumonix, the Rat King
1 Lightning Greaves
1 Locust Miser
1 Lord Skitter, Sewer King
1 Lord Skitter's Butcher
1 Luka, the Traveling Sound
1 Midgar, City of Mako
1 Mirkwood Bats
1 Morbid Opportunist
1 Mudflat Village
1 Myriad Landscape
1 Nezumi Informant
1 Okiba-Gang Shinobi
1 Pack Rat
1 Path of Ancestry
1 Pestilence Rats
1 Peter Parker's Camera
1 Piper of the Swarm
1 Pitiless Plunderer
1 Plumb the Forbidden
1 Rancid Rats
1 Rat King, Pale Piper
1 Rat King, Verminister
1 Reliquary Tower
1 Saw in Half
1 Sign in Blood
1 Skullclamp
1 Sol Ring
1 Species Specialist
1 Susur Secundi, Void Altar
1 Swamp
1 Swamp
1 Swamp
1 Swamp
1 Swamp
1 Swamp
1 Swamp
1 Swamp
2 Swamp
3 Swamp
2 Swamp
1 Swamp
3 Swamp
1 Swamp
1 Swamp
1 Swarm of Rats
1 Swarmyard
1 Swarmyard Massacre
1 Swiftfoot Boots
1 Tangled Colony
1 Temple of the False God
1 The Meathook Massacre
1 Thornbite Staff
1 Throat Slitter
1 Typhoid Rats
1 V.A.T.S.
1 Valley Rotcaller
1 Village Rites
1 Voracious Vermin
1 War Room
1 Witch's Cottage
1 Zulaport Cutthroat
1 Marrow-Gnawer
    """
    evaluar_mazo(mi_mazo, model_path=MODEL_PATH)