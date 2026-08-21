import os
import torch
import torch.nn.functional as F

from deck_parser import parse_moxfield_deck
from deck_graph_builder import DeckGraphBuilder
from edh_gnn_model import EDHPowerGNN
from mapeo_nlp import extraer_hallazgos_28d, construir_reporte_llm


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


def evaluar_mazo_api(
    deck_text: str, 
    model=None, 
    builder=None, 
    device: str = "cpu", 
    nombre_mazo: str = "Commander Deck",
    model_path: str = MODEL_PATH,
    **kwargs
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # 1. Parsear cartas
    raw_cards = parse_moxfield_deck(deck_text)
    
    # 2. Construir Grafo (Reutilizando el builder si viene desde FastAPI)
    if builder is None:
        builder = DeckGraphBuilder()
        
    x_target, edge_target = builder.build_graph_from_decklist(raw_cards)
    
    if x_target.size(0) == 0:
        return {"error": "No se pudieron extraer nodos del mazo."}

    x_target = x_target.to(device)
    edge_target = edge_target.to(device)

    # Conteo de Game Changers (Dimensión 28 -> índice 27)
    game_changer_flags = x_target[:, 27]
    gc_count = int(torch.sum(game_changer_flags).item())

    # 3. Cargar o Reutilizar el Modelo GNN
    if model is None:
        model = EDHPowerGNN(in_channels=28, hidden_channels=64, num_classes=5).to(device)
        if model_path and os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            except Exception as e:
                print(f"⚠️ Error al cargar weights: {e}")
    else:
        model = model.to(device)

    # 4. Inferencia
    batch_vector = torch.zeros(x_target.size(0), dtype=torch.long, device=device)
    model.eval()
    
    with torch.no_grad():
        out = model(x_target, edge_target, batch_vector)
        temperatura = 0.3  
        probs = F.softmax(out / temperatura, dim=1).cpu().numpy()[0]

    # 5. Generación de Explicabilidad y Reporte
    predicted_bracket = int(probs.argmax())
    hallazgos = extraer_hallazgos_28d(x_target, raw_cards)
    reporte_ia = construir_reporte_llm(probs, hallazgos, nombre_mazo=nombre_mazo)

    # 6. Retorno estructurado en formato JSON
    return {
        "predicted_bracket": BRACKET_NAMES[predicted_bracket],
        "bracket_index": predicted_bracket + 1,
        "probabilities": {
            BRACKET_NAMES[i]: round(float(p * 100), 2) for i, p in enumerate(probs)
        },
        "game_changers_count": gc_count,
        "processed_cards": len(raw_cards),
        "report": reporte_ia
    }
