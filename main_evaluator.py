# main_evaluator.py
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


def evaluar_mazo_api(deck_text: str, model: torch.nn.Module, builder: DeckGraphBuilder, device: str = "cpu", nombre_mazo: str = "Mazo Analizado") -> dict:
    raw_cards = parse_moxfield_deck(deck_text)
    
    # 1. CONSTRUCCIÓN DEL GRAFO DESDE LA BASE DE DATOS
    x_target, edge_target = builder.build_graph_from_decklist(raw_cards)
    
    if x_target.size(0) == 0:
        raise ValueError("No se pudieron extraer nodos o cartas válidas del mazo.")

    # Mover tensores al dispositivo objetivo (CPU/GPU/MPS)
    x_target = x_target.to(device)
    edge_target = edge_target.to(device)

    # Contar Game Changers automáticamente desde el vector de 28D (Índice 27)
    game_changer_flags = x_target[:, 27]
    gc_count = int(torch.sum(game_changer_flags).item())

    # 2. CREAR TENSOR DE BATCH
    batch_vector = torch.zeros(x_target.size(0), dtype=torch.long, device=device)

    # 3. INFERENCIA PURA CON TEMPERATURA
    model.eval()
    with torch.no_grad():
        out = model(x_target, edge_target, batch_vector)
        
        # Factor de temperatura: 0.6 para afilar el pico de mayor confianza
        temperatura = 0.6  
        probs = F.softmax(out / temperatura, dim=1).cpu().numpy()[0]

    # 4. DIAGNÓSTICO Y EXPLICABILIDAD VECTORIAL DE LA GNN (28D)
    feature_activations = torch.sum(x_target, dim=0).cpu().numpy()
    valid_activations = feature_activations[:len(FEATURE_NAMES)]
    top_indices = valid_activations.argsort()[::-1][:5]
    
    top_features = []
    for idx in top_indices:
        nombre_feat = FEATURE_NAMES[idx]
        activacion = float(valid_activations[idx])
        top_features.append({"feature": nombre_feat, "density": round(activacion, 1)})

    # 5. REPORTE NLP DE 28 DIMENSIONES
    try:
        hallazgos = extraer_hallazgos_28d(x_target, raw_cards)
        reporte_texto = construir_reporte_llm(probs, hallazgos, nombre_mazo=nombre_mazo)
    except Exception as e:
        print(f"⚠️ Error al generar el reporte NLP: {e}")
        reporte_texto = "No se pudo generar el reporte detallado."

    predicted_bracket = int(probs.argmax())

    # 6. RETORNO ESTRUCTURADO PARA FASTAPI
    return {
        "cards_processed": len(raw_cards),
        "nodes_built": int(x_target.size(0)),
        "game_changers_count": gc_count,
        "predicted_bracket_id": predicted_bracket + 1,
        "predicted_bracket_name": BRACKET_NAMES[predicted_bracket],
        "probabilities": {BRACKET_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(probs)},
        "top_features": top_features,
        "report": reporte_texto
    }
