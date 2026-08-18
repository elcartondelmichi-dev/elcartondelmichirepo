import os
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from curl_cffi import requests
import torch

from edh_gnn_model import EDHPowerGNN
from deck_graph_builder import DeckGraphBuilder
from main_evaluator import evaluar_mazo_api

ml_assets = {}

def extraer_texto_desde_moxfield_json(deck_data: dict) -> str:
    """Extrae comandantes y mainboard soportando todas las estructuras de Moxfield v2."""
    lines = []
    
    # 1. Intentar obtener 'boards' o buscar en la raíz del JSON
    boards = deck_data.get("boards", {})
    
    # Buscar el mainboard en las distintas rutas posibles
    mainboard_data = (
        boards.get("mainboard", {}) or 
        deck_data.get("mainboard", {})
    )
    
    # Buscar el commander en las distintas rutas posibles (plural y singular)
    commander_data = (
        boards.get("commanders", {}) or 
        boards.get("commander", {}) or 
        deck_data.get("commanders", {}) or 
        deck_data.get("commander", {})
    )
    
    # Extraer los diccionarios de cartas
    mainboard_cards = mainboard_data.get("cards", {}) if isinstance(mainboard_data, dict) else {}
    commander_cards = commander_data.get("cards", {}) if isinstance(commander_data, dict) else {}
    
    # Unir ambas secciones
    all_cards = {**commander_cards, **mainboard_cards}
    
    # Recorrer cartas
    for key, details in all_cards.items():
        if not isinstance(details, dict):
            continue
            
        quantity = details.get("quantity", 1)
        
        # Moxfield guarda el nombre real en details["card"]["name"] o en la llave 'key'
        card_obj = details.get("card", {})
        if isinstance(card_obj, dict) and card_obj.get("name"):
            card_name = card_obj.get("name")
        else:
            card_name = key
            
        lines.append(f"{quantity}x {card_name}")
        
    return "\n".join(lines)
@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cpu"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "edh_gnn_model.pt")
    
    builder = DeckGraphBuilder()
    
    # Obtener dinámicamente la dimensión real del extractor para no hardcodear 26
    sample_vec = builder.extractor.extract_features("Sol Ring")
    in_channels = len(sample_vec) if sample_vec is not None else 23
    
    # Instanciar arquitectura con las dimensiones reales del extractor
    model = EDHPowerGNN(in_channels=in_channels, hidden_channels=64, num_classes=5).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    ml_assets["model"] = model
    ml_assets["builder"] = builder
    ml_assets["device"] = device
    
    yield
    ml_assets.clear()

app = FastAPI(lifespan=lifespan)

class DeckRequest(BaseModel):
    deck_url: str

@app.post("/predict")
def predict_power_level(request: DeckRequest):
    match = re.search(r'decks/([a-zA-Z0-9_-]+)', request.deck_url)
    if not match:
        raise HTTPException(status_code=400, detail="URL de Moxfield inválida")
    
    deck_id = match.group(1)
    moxfield_api = f"https://api.moxfield.com/v2/decks/all/{deck_id}"
    
    try:
        response = requests.get(moxfield_api, impersonate="chrome", timeout=10)
        if response.status_code != 200:
            return {"success": False, "error": f"Moxfield devolvió HTTP {response.status_code}"}
        
        deck_data = response.json()
        deck_text = extraer_texto_desde_moxfield_json(deck_data)
        
        if not deck_text.strip():
            return {"success": False, "error": "No se pudieron extraer cartas del JSON de Moxfield."}
        
        resultado_evaluacion = evaluar_mazo_api(
            deck_text=deck_text,
            model=ml_assets["model"],
            builder=ml_assets["builder"],
            device=ml_assets["device"]
        )
        
        return {
            "success": True,
            "deck_name": deck_data.get("name", "Commander Deck"),
            "evaluation": resultado_evaluacion
        }
        
    except Exception as e:
        return {"success": False, "error": f"Error procesando el mazo: {str(e)}"}
