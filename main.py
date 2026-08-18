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
    lines = []
    
    def procesar_seccion_cartas(seccion):
        if not seccion:
            return
            
        if isinstance(seccion, dict):
            cartas = seccion.get("cards", seccion)
            if isinstance(cartas, dict):
                for key, details in cartas.items():
                    if isinstance(details, dict):
                        qty = details.get("quantity", 1)
                        card_obj = details.get("card", {})
                        name = card_obj.get("name") if isinstance(card_obj, dict) and card_obj.get("name") else key
                        lines.append(f"{qty} {name}")  # <--- SIN LA "x", SOLO ESPACIO
            elif isinstance(cartas, list):
                for item in cartas:
                    if isinstance(item, dict):
                        qty = item.get("quantity", 1)
                        card_obj = item.get("card", {})
                        name = card_obj.get("name", "Unknown") if isinstance(card_obj, dict) else item.get("name", "Unknown")
                        lines.append(f"{qty} {name}")  # <--- SIN LA "x", SOLO ESPACIO

    boards = deck_data.get("boards", {})
    if isinstance(boards, dict):
        for board_name, board_content in boards.items():
            procesar_seccion_cartas(board_content)

    for root_key in ["mainboard", "commanders", "commander", "deck"]:
        if root_key in deck_data:
            procesar_seccion_cartas(deck_data[root_key])

    return "\n".join(lines)
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cpu"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "edh_gnn_model.pt")
    
    builder = DeckGraphBuilder()
    
    # Obtener la dimensión real del extractor
    sample_vec = builder.extractor.extract_features("Sol Ring")
    in_channels = len(sample_vec) if sample_vec is not None else 23
    
    # Instanciar el modelo con PyTorch
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
            return {
                "success": False, 
                "error": "No se pudieron extraer cartas del JSON de Moxfield."
            }
        
        # Inferencia con la GNN
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
