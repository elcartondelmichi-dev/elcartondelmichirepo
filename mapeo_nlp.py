# mapeo_nlp.py

def obtener_cartas_clave_por_feature(x_target, raw_cards):
    """
    Mapea las 23 dimensiones exactas de CardFeatureExtractor a las cartas correspondientes.
    """
    # Mapeo alineado 1:1 con el vector de 23 posiciones
    feature_map = {
        13: "Aceleración / Ramp",      # _check_ramp
        14: "Motores de Robo",         # _check_card_draw
        15: "Tutores y Búsquedas",     # search library (no lands)
        16: "Interacción / Removal",   # _check_removal
        17: "Contrahechizos / Counters",# counter target spell
        18: "Limpiamesas / Board Wipes",# destroy all / exile all
        19: "Turnos Extra",            # extra turn
        20: "Destrucción de Tierras",  # MLD
        21: "Copia / Storm",           # _check_spell_copy
        22: "Game Changers"            # Flag especial de la DB
    }

    hallazgos = {}

    for feat_idx, feat_name in feature_map.items():
        # Buscamos qué nodos activan la característica correspondiente
        card_indices = (x_target[:, feat_idx] > 0).nonzero(as_tuple=True)[0]

        if len(card_indices) > 0:
            names = [raw_cards[i] for i in card_indices.cpu().numpy()]
            
            # Filtramos tierras básicas por si colisionan con alguna regla de texto
            names = [
                n for n in names 
                if n not in ["Forest", "Island", "Mountain", "Swamp", "Plains"]
            ]

            if names:
                hallazgos[feat_name] = names[:4]  # Muestra máximo 4 ejemplos por categoría

    return hallazgos


def construir_reporte_humano(probs, hallazgos):
    """
    Genera el diagnóstico en lenguaje natural a partir de las características reales.
    """
    lineas = []
    lineas.append("🗣️ DIAGNÓSTICO ESTRUCTURAL (GNN -> NLP Local):")
    lineas.append("-" * 65)

    if "Game Changers" in hallazgos:
        lineas.append(f"  • ⚠️ Game Changers clave: {', '.join(hallazgos['Game Changers'])}.")

    if "Aceleración / Ramp" in hallazgos:
        lineas.append(f"  • ⚡ Aceleración y Maná: Fuentes encontradas como {', '.join(hallazgos['Aceleración / Ramp'])}.")

    if "Motores de Robo" in hallazgos:
        lineas.append(f"  • 📚 Ventaja de Cartas: Motores de robo impulsados por {', '.join(hallazgos['Motores de Robo'])}.")

    if "Tutores y Búsquedas" in hallazgos:
        lineas.append(f"  • 🔍 Consistencia y Búsquedas: Filtrado apoyado en {', '.join(hallazgos['Tutores y Búsquedas'])}.")

    if "Interacción / Removal" in hallazgos:
        lineas.append(f"  • 🎯 Removal Dirigido: Interacción basada en {', '.join(hallazgos['Interacción / Removal'])}.")

    if "Contrahechizos / Counters" in hallazgos:
        lineas.append(f"  • 🛡️ Respuesta e Interrupción: Pila defendida por {', '.join(hallazgos['Contrahechizos / Counters'])}.")

    if "Limpiamesas / Board Wipes" in hallazgos:
        lineas.append(f"  • 🧹 Control de Mesa: Respuestas masivas detectadas en {', '.join(hallazgos['Limpiamesas / Board Wipes'])}.")

    if "Copia / Storm" in hallazgos:
        lineas.append(f"  • 🔄 Copia y Multiplicadores: Sinergia de hechizos en {', '.join(hallazgos['Copia / Storm'])}.")

    if "Turnos Extra" in hallazgos:
        lineas.append(f"  • ⏳ Modificadores de Tiempo: Turnos extra con {', '.join(hallazgos['Turnos Extra'])}.")

    if "Destrucción de Tierras" in hallazgos:
        lineas.append(f"  • 🛑 Denegación de Recursos (MLD): Amenazas en {', '.join(hallazgos['Destrucción de Tierras'])}.")

    return "\n".join(lineas)