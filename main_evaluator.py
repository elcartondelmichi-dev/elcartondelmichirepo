# main_evaluator.py
import os
import torch
import torch.nn.functional as F

from deck_parser import parse_moxfield_deck
from deck_graph_builder import DeckGraphBuilder
from mapeo_nlp import obtener_cartas_clave_por_feature, construir_reporte_humano

BRACKET_NAMES = [
    "Bracket 1: Exhibition (Ultra-Casual)",
    "Bracket 2: Core (Precons)",
    "Bracket 3: Upgraded (Mid-High Power)",
    "Bracket 4: Optimized (High Power)",
    "Bracket 5: cEDH (Competitive Metagame)"
]

FEATURE_NAMES = [
    "CMC / Curva de Maná", "Es Criatura", "Es Artefacto", "Es Encantamiento",
    "Es Conjuro/Instante", "Es Tierra", "Produce Maná", "Es Tutor/Búsqueda",
    "Es Board Wipe/Removal", "Es Motor de Robo", "Efecto Untap/Sinergia", 
    "Potencial de Combo", "Fast Mana", "Interacción Barata", "Sinergia Tribal", 
    "Contadores / Sinergias", "Poder/Resistencia", "Incoloro", "Blanco", 
    "Azul", "Negro", "Rojo", "Game Changer Flag"
]

def evaluar_mazo_api(deck_text: str, model: torch.nn.Module, builder: DeckGraphBuilder, device: str = "cpu") -> dict:
    # 1. Parsear cartas
    raw_cards = parse_moxfield_deck(deck_text)
    
    # TRACE 1: Verificar si parse_moxfield_deck devolvió cartas
    print(f"[DEBUG EVALUATOR] Total cartas parseadas: {len(raw_cards)}")
    if raw_cards:
        print(f"[DEBUG EVALUATOR] Muestra primeras 3 cartas: {raw_cards[:3]}")
    
    # 2. Construir el grafo
    x_target, edge_target = builder.build_graph_from_decklist(raw_cards)
    
    # TRACE 2: Verificar dimensión del tensor resultante
    print(f"[DEBUG EVALUATOR] Nodos creados en x_target: {x_target.size(0)}")
    
    if x_target.size(0) == 0:
        raise ValueError(
            f"No se pudieron extraer nodos o cartas válidas del mazo. "
            f"(Cartas recibidas: {len(raw_cards)}, Nodos construidos: {x_target.size(0)})"
        )

    # 3. Preparar tensores para PyTorch Geometric
    batch = torch.zeros(x_target.size(0), dtype=torch.long).to(device)
    x_target = x_target.to(device)
    edge_target = edge_target.to(device)

    # 4. Inferencia
    model.eval()
    with torch.no_grad():
        logits = model(x_target, edge_target, batch)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    # 5. Mapeo NLP y conteos
    try:
        hallazgos = obtener_cartas_clave_por_feature(x_target, raw_cards)
        reporte_texto = construir_reporte_humano(probs, hallazgos)
    except Exception as e:
        print(f"⚠️ Error generando reporte NLP: {e}")
        reporte_texto = "No se pudo generar el reporte detallado."

    # Detectar Game Changers (asumiendo que es el último feature de tu vector)
    gc_count = int(x_target[:, -1].sum().item()) if x_target.size(1) >= len(FEATURE_NAMES) else 0

    predicted_bracket = int(probs.argmax())

    # 6. Retorno obligatorio para la API de FastAPI
    return {
        "cards_processed": len(raw_cards),
        "nodes_built": int(x_target.size(0)),
        "game_changers_count": gc_count,
        "predicted_bracket_id": predicted_bracket + 1,
        "predicted_bracket_name": BRACKET_NAMES[predicted_bracket],
        "probabilities": {BRACKET_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(probs)},
        "report": reporte_texto
    }
