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
