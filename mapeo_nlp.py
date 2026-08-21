# mapeo_nlp.py
import numpy as np

# Mapeo alineado 1:1 con las 28 posiciones de CardFeatureExtractor
FEATURE_MAP = {
    13: ("Aceleración y Ramp Tradicional", "⚡", "ramp"),
    14: ("Fast Mana / Rituals de Aceleración", "💎", "fast_mana"),
    15: ("Ventaja de Cartas / Robo", "📚", "card_advantage"),
    16: ("Tutores Eficientes (CMC ≤ 2)", "🔍", "tutors_low"),
    17: ("Tutores Estándar / Flexibles (CMC ≥ 3)", "📖", "tutors_high"),
    18: ("Removal Dirigido / Interacción", "🎯", "removal"),
    19: ("Contrahechizos / Protección en Pila", "🛡️", "counterspells"),
    20: ("Limpiamesas / Board Wipes", "🧹", "wipes"),
    21: ("Modificadores de Tiempo / Extra Turns", "⏳", "extra_turns"),
    22: ("Denegación de Recursos (MLD)", "🛑", "mld"),
    23: ("Copia de Hechizos / Storm", "🔄", "storm"),
    24: ("Reanimación / Trampa de Maná (Cheat Mana)", "💀", "cheat_mana"),
    25: ("Efectos de Drenaje / Burn / Pingers", "🩸", "pingers"),
    26: ("Multiplicadores de Disparos / Ataques Extra", "💥", "triggers"),
    27: ("Game Changers / Staples de Alto Impacto", "⚠️", "game_changers")
}

BRACKET_INFO = {
    1: ("Exhibition (Ultra-Casual)", "un nivel totalmente casual enfocado en temática o curvas pesadas, sin intenciones explosivas."),
    2: ("Core (Precons / Casual Standard)", "un nivel equivalente a Preconstruidos oficiales con estructura sólida pero velocidad moderada."),
    3: ("Upgraded (Mid-High Power)", "un nivel optimizado con alta sinergia, interacción constante y motores de victoria sólidos sin llegar al exceso de velocidad de cEDH."),
    4: ("Optimized (High Power)", "un entorno agresivo de alta velocidad, tutoría rápida y consistencia para cerrar partidas en turnos tempranos."),
    5: ("cEDH (Competitive Metagame)", "la cúspide del formato. Máxima velocidad, Fast Mana agresivo, respuestas a costo 0/1 y combos de turno 1-3.")
}

BASIC_LANDS = {
    "Forest", "Island", "Mountain", "Swamp", "Plains", 
    "Snow-Covered Forest", "Snow-Covered Island", "Snow-Covered Mountain", 
    "Snow-Covered Swamp", "Snow-Covered Plains"
}


def extraer_hallazgos_28d(x_target, raw_cards):
    """
    Escanea el tensor 28D y extrae las cartas activadas por categoría.
    """
    hallazgos = {}

    if hasattr(x_target, "cpu"):
        x_target_np = x_target.cpu().numpy()
    else:
        x_target_np = np.array(x_target)

    for feat_idx, (feat_name, emoji, tag) in FEATURE_MAP.items():
        if feat_idx >= x_target_np.shape[1]:
            continue

        card_indices = np.where(x_target_np[:, feat_idx] > 0)[0]
        if len(card_indices) > 0:
            names = [raw_cards[i] for i in card_indices if raw_cards[i] not in BASIC_LANDS]
            if names:
                # Muestra variada (hasta 4 cartas)
                sample = names[:4] if len(names) <= 4 else list(np.random.choice(names, 4, replace=False))
                hallazgos[feat_name] = {
                    "emoji": emoji,
                    "tag": tag,
                    "cards": sample,
                    "total": len(names)
                }

    return hallazgos


def construir_reporte_llm(probs, hallazgos, nombre_mazo="Mazo Evaluado", **kwargs):
    """
    Motor NLP Analítico de Inferencia Rápida.
    Genera diagnósticos estructurados y matizados según la certeza probabilística de la GNN.
    """
    sorted_indices = np.argsort(probs)[::-1]
    bracket_idx = int(sorted_indices[0]) + 1
    confianza = float(probs[bracket_idx - 1]) * 100
    
    second_idx = int(sorted_indices[1]) + 1
    second_prob = float(probs[second_idx - 1]) * 100

    nombre_b, desc_b = BRACKET_INFO[bracket_idx]

    lineas = []
    lineas.append("🧠 ================= DIAGNÓSTICO EVALUATIVO (NLP) =================")
    lineas.append(f"🎯 **DIAGNÓSTICO PRINCIPAL:** {nombre_b} (Bracket {bracket_idx})")
    lineas.append(f"📊 **Nivel de Certeza de la Red:** {confianza:.1f}%\n")

    # 1. Análisis de Certeza y Contexto (Rule 0)
    lineas.append("📝 **Resumen Evaluativo:**")
    lineas.append(f"La GNN clasifica a **{nombre_mazo}** dentro del **Bracket {bracket_idx}** ({confianza:.1f}% de certeza), {desc_b}")
    
    if confianza < 65.0 or (confianza - second_prob < 20.0):
        lineas.append(f"\n⚠️ **Advertencia de Rule 0 (Mazo Borde):**")
        lineas.append(f"El mazo se encuentra en la frontera entre **Bracket {bracket_idx}** ({confianza:.1f}%) y **Bracket {second_idx}** ({second_prob:.1f}%). Se recomienda discutir en la mesa el ritmo del mazo antes de jugar.")
    elif second_prob > 12.0:
        lineas.append(f"\n💡 *Inclinación Secundaria:* Muestra trazas del **Bracket {second_idx}** ({second_prob:.1f}%) debido a la aceleración o densidad de sinergia puntual.")

    # 2. Pilares Estructurales
    lineas.append("\n📌 **Pilares Estructurales Detectados en el Grafo:**")

    prioridades = [
        "Game Changers / Staples de Alto Impacto",
        "Fast Mana / Rituals de Aceleración",
        "Tutores Eficientes (CMC ≤ 2)",
        "Reanimación / Trampa de Maná (Cheat Mana)",
        "Ventaja de Cartas / Robo",
        "Removal Dirigido / Interacción"
    ]

    mostrados = 0
    for prio in prioridades:
        if prio in hallazgos:
            info = hallazgos[prio]
            cartas_str = ", ".join(f"`{c}`" for c in info["cards"])
            lineas.append(f"  {info['emoji']} **{prio}:** {cartas_str} ({info['total']} detectadas)")
            mostrados += 1

    for feat_name, info in hallazgos.items():
        if feat_name not in prioridades and mostrados < 8:
            cartas_str = ", ".join(f"`{c}`" for c in info["cards"])
            lineas.append(f"  {info['emoji']} **{feat_name}:** {cartas_str} ({info['total']} detectadas)")
            mostrados += 1

    # 3. Puntos Ciegos
    carencias = []
    if "Ventaja de Cartas / Robo" not in hallazgos or hallazgos["Ventaja de Cartas / Robo"]["total"] < 3:
        carencias.append("Escaso motor de robo / ventaja de cartas")
    if "Removal Dirigido / Interacción" not in hallazgos and "Contrahechizos / Protección en Pila" not in hallazgos:
        carencias.append("Baja interacción/protección inmediata")

    if carencias:
        lineas.append(f"\n🔍 **Puntos Ciegos Detectados:** {', '.join(carencias)}.")

    # 4. Dictamen de Mesa
    lineas.append("\n⚙️ **Dictamen Evaluativo de Mesa:**")
    if bracket_idx == 5:
        lineas.append("🔥 **Status cEDH:** Lista de máxima aceleración e interacción. Diseñada para ganar en turnos 1-3 o cerrar la mesa con combos consistentes.")
    elif bracket_idx == 4:
        lineas.append("⚡ **Status High Power:** Mazo muy veloz y consistente. Apto para mesas competitivas sin llegar al meta de cEDH puro.")
    elif bracket_idx == 3:
        lineas.append("🛡️ **Status Mid-High:** Presenta una sinergia interna sólida y excelente valor en mesa. Capaz de cerrar partidas con potencia sin romper el formato casual.")
    elif bracket_idx == 2:
        lineas.append("🎲 **Status Precon/Casual:** Mazo bien estructurado pero con limitantes de velocidad, falta de tutoría agresiva o dependencia crítica de la zona de comando.")
    else:
        lineas.append("🌱 **Status Ultra-Casual:** Mazo guiado por temática o curva alta. Pensado para partidas largas de interacción relajada.")

    lineas.append("==================================================================")
    return "\n".join(lineas)


def obtener_cartas_clave_por_feature(x_target, raw_cards):
    """
    Función de compatibilidad local: Mapea las dimensiones de 28D a listas simples de cartas.
    """
    hallazgos_28d = extraer_hallazgos_28d(x_target, raw_cards)
    hallazgos_simples = {}

    for feat_name, data in hallazgos_28d.items():
        hallazgos_simples[feat_name] = data["cards"]

    return hallazgos_simples


def construir_reporte_humano(probs, hallazgos):
    """
    Genera el diagnóstico en lenguaje natural a partir de un diccionario simple de hallazgos.
    """
    lineas = []
    lineas.append("🗣️ DIAGNÓSTICO ESTRUCTURAL (GNN -> NLP Local):")
    lineas.append("-" * 65)

    for feat_name, info in hallazgos.items():
        # Soporta diccionario estructurado o lista simple de cartas
        if isinstance(info, dict):
            cartas = info.get("cards", [])
            emoji = info.get("emoji", "•")
        else:
            cartas = info
            emoji = "•"

        if cartas:
            cartas_str = ", ".join(cartas)
            lineas.append(f"  {emoji} **{feat_name}:** {cartas_str}.")

    return "\n".join(lineas)


# Aliases para mantener compatibilidad con versiones anteriores o imports dinámicos
construir_reporte_gemini = construir_reporte_llm
