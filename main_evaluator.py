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
    
    # 2. Construir el grafo
    x_target, edge_target = builder.build_graph_from_decklist(raw_cards)
    
    if x_target.size(0) == 0:
        raise ValueError("No se pudieron extraer nodos o cartas válidas del mazo.")

    x_target = x_target.to(device)
    edge_target = edge_target.to(device)

    # Conteo de Game Changers (Dimensión 23 -> índice 22)
    game_changer_flags = x_target[:, 22]
    gc_count = int(torch.sum(game_changer_flags).item())

    # 3. Batch Vector e Inferencia
    batch_vector = torch.zeros(x_target.size(0), dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(x_target, edge_target, batch_vector)
        temperatura = 0.6  
        probs = F.softmax(out / temperatura, dim=1).cpu().numpy()[0]

    # 4. Diagnóstico de características más activas
    feature_activations = torch.sum(x_target, dim=0).cpu().numpy()
    top_indices = feature_activations.argsort()[::-1][:5]

    top_features = []
    for idx in top_indices:
        nombre_feat = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"Feature #{idx}"
        top_features.append({
            "feature": nombre_feat,
            "density": float(feature_activations[idx])
        })

    # 5. Reporte NLP
    hallazgos = obtener_cartas_clave_por_feature(x_target, raw_cards)
    reporte_texto = construir_reporte_humano(probs, hallazgos)

    predicted_bracket = int(probs.argmax())

    # Retorno serializable para FastAPI
    return {
        "cards_processed": len(raw_cards),
        "game_changers_count": gc_count,
        "predicted_bracket_id": predicted_bracket + 1,
        "predicted_bracket_name": BRACKET_NAMES[predicted_bracket],
        "probabilities": {BRACKET_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(probs)},
        "top_features": top_features,
        "report": reporte_texto
    }
