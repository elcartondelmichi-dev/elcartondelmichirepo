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

# Estado global en memoria
ml_assets = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cpu"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "edh_gnn_model.pt")
    
    # Instanciar arquitectura y cargar pesos
    model = EDHPowerGNN(in_channels=26, hidden_channels=64, num_classes=5).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Instanciar builder (se conecta a la DB internamente)
    builder = DeckGraphBuilder()
    
    ml_assets["model"] = model
    ml_assets["builder"] = builder
    ml_assets["device"] = device
    
    yield
    ml_assets.clear()

app = FastAPI(lifespan=lifespan)

class DeckRequest(BaseModel):
    deck_url: str

def extraer_texto_desde_moxfield_json(deck_data: dict) -> str:
    """Convierte el JSON de la API de Moxfield al formato de texto '1x Nombre de Carta'"""
    lines = []
    boards = deck_data.get("boards", {})
    mainboard = boards.get("mainboard", {}).get("cards", {})
    commanders = boards.get("commander", {}).get("cards", {})
    
    # Unir Commander + Mainboard
    all_cards = {**commanders, **mainboard}
    
    for card_name, details in all_cards.items():
        quantity = details.get("quantity", 1)
        lines.append(f"{quantity} {card_name}")
        
    return "\n".join(lines)

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
        
        # 1. Convertir JSON de Moxfield a texto de lista
        deck_text = extraer_texto_desde_moxfield_json(deck_data)
        
        # 2. Ejecutar Inferencia GNN
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
