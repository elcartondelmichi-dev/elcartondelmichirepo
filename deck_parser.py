import re

def parse_moxfield_deck(deck_text: str) -> list[str]:
    """
    Toma un string copia-pegado de Moxfield/MTGO/Arena y devuelve 
    la lista aplanada con los nombres limpios de las cartas (100 en total).
    """
    cards = []
    lines = deck_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        # Ignora líneas vacías, comentarios o encabezados de sección de Moxfield (Ej: SIDEBOARD:, COMMANDER:)
        if not line or line.startswith("//") or line.endswith(":"):
            continue
            
        # Capturamos la cantidad (grupo 1) y el resto de la línea (grupo 2)
        match = re.match(r'^(\d+)\s+(.*)$', line)
        if match:
            quantity = int(match.group(1))
            raw_card_name = match.group(2).strip()
            
            # Limpia metadatos opcionales de Moxfield como (SET) 123 *F* o [SET]
            clean_name = re.sub(r'\s*(\([A-Z0-9]+\)|\[[A-Z0-9]+\]|\*\w+\*).*$', '', raw_card_name).strip()
            
            # Agregamos la carta tantas veces como indique la cantidad
            cards.extend([clean_name] * quantity)
        else:
            # Si no tiene número al inicio, se asume cantidad 1
            clean_name = re.sub(r'\s*(\([A-Z0-9]+\)|\[[A-Z0-9]+\]|\*\w+\*).*$', '', line).strip()
            if clean_name:
                cards.append(clean_name)
                
    return cards


if __name__ == "__main__":
    # Prueba rápida de sanitización
    test_mox = """
    // Decklist Export
    1 Shorikai, Genesis Engine (NEC) 42 *F*
    1 Sol Ring [LTC]
    2 Island
    """
    parsed = parse_moxfield_deck(test_mox)
    print(f"Cartas parseadas ({len(parsed)}):", parsed)
