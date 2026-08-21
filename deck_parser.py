import re

def parse_moxfield_deck(deck_text: str) -> list[str]:
    """
    Toma un string copia-pegado de Moxfield y devuelve la lista completa de cartas (100 en total).
    """
    cards = []
    lines = deck_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("//"):
            continue
            
        # Capturamos la cantidad (grupo 1) y el nombre (grupo 2)
        match = re.match(r'^(\d+)\s+(.*)$', line)
        if match:
            quantity = int(match.group(1))
            card_name = match.group(2).strip()
            
            # Agregamos la carta tantas veces como indique la cantidad
            cards.extend([card_name] * quantity)
        else:
            cards.append(line)
                
    return cards