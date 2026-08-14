from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from curl_cffi import requests
import re

app = FastAPI()

class DeckRequest(BaseModel):
    deck_url: str

@app.post("/predict")
def predict_power_level(request: DeckRequest):
    # 1. Extraer ID del mazo
    match = re.search(r'decks/([a-zA-Z0-9_-]+)', request.deck_url)
    if not match:
        raise HTTPException(status_code=400, detail="URL de Moxfield inválida")
    
    deck_id = match.group(1)
    
    # 2. Consultar Moxfield impersonando a Chrome (Bypassea Cloudflare)
    moxfield_api = f"https://api.moxfield.com/v2/decks/all/{deck_id}"
    
    try:
        # impersonate="chrome" emula la firma TLS exacta de Chrome
        response = requests.get(moxfield_api, impersonate="chrome", timeout=10)
        
        if response.status_code != 200:
            return {"success": False, "error": f"Moxfield devolvió HTTP {response.status_code}"}
        
        deck_data = response.json()
        
        # 3. Extraer cartas y armar la matriz de adyacencia
        # ... Aquí corre la magia de tu GNN de 28 nodos ...
        
        # Ejemplo de retorno
        return {
            "success": True,
            "power_level": 8.5,  # Resultado de tu GNN
            "deck_name": deck_data.get("name", "Commander Deck")
        }
        
    except Exception as e:
        return {"success": False, "error": f"Error procesando el mazo: {str(e)}"}
