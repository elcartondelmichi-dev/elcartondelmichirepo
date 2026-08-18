import sqlite3
import random
import torch
from torch_geometric.data import Data

def generar_mazo_dinamico_por_bracket(db_path, bracket_objetivo):
    """
    Toma cartas de la BD sqlite basándose en estadísticas reales (CMC, Game Changers, Tipos)
    en lugar de listas estáticas.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    deck_cards = []
    
    if bracket_objetivo == 1: # Precon / Casual Bajo
        # Cartas de CMC medio-alto, sin game changers
        cursor.execute("SELECT name FROM cards WHERE is_game_changer = 0 AND cmc >= 3 ORDER BY RANDOM() LIMIT 40")
        deck_cards.extend([row[0] for row in cursor.fetchall()])
        # Tierras básicas
        deck_cards.extend(["Forest"] * 20 + ["Swamp"] * 20 + ["Island"] * 20)

    elif bracket_objetivo == 3: # Sinergia Alta (Ej. Elfos/Tribal)
        # Buscar cartas con tipo 'Elf' o sinérgicas
        cursor.execute("SELECT name FROM cards WHERE type_line LIKE '%Elf%' ORDER BY RANDOM() LIMIT 30")
        deck_cards.extend([row[0] for row in cursor.fetchall()])
        # Relleno de curva baja
        cursor.execute("SELECT name FROM cards WHERE cmc <= 2 ORDER BY RANDOM() LIMIT 30")
        deck_cards.extend([row[0] for row in cursor.fetchall()])
        # Tierras
        deck_cards.extend(["Forest"] * 25 + ["Overgrown Tomb"] * 15)

    elif bracket_objetivo == 5: # cEDH / Max Power
        # Cartas con game-changers, tutores, CMC ultra bajo
        cursor.execute("SELECT name FROM cards WHERE is_game_changer = 1 ORDER BY RANDOM() LIMIT 10")
        deck_cards.extend([row[0] for row in cursor.fetchall()])
        cursor.execute("SELECT name FROM cards WHERE cmc <= 1 ORDER BY RANDOM() LIMIT 50")
        deck_cards.extend([row[0] for row in cursor.fetchall()])
        deck_cards.extend(["Command Tower", "City of Brass", "Mana Confluence"] * 10)

    conn.close()
    
    # Rellenar o recortar a exactamente 100 cartas
    while len(deck_cards) < 100:
        deck_cards.append("Forest")
    return deck_cards[:100]