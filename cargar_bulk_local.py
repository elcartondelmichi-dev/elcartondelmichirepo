import json
import sqlite3
import os

DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))

# Nombre del archivo con la extensión .json / .jsonl
JSON_BULK_FILE = os.path.join(DIRECTORIO_ACTUAL, "all-cards.jsonl")
DB_FILE = os.path.join(DIRECTORIO_ACTUAL, "prueba.db")

print(f"Buscando archivo en: {JSON_BULK_FILE}")

def poblar_db_desde_jsonl_local():
    if not os.path.exists(JSON_BULK_FILE):
        print(f"[x] Error: No encontré el archivo {JSON_BULK_FILE}. Revisa la ruta.")
        return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 1. Crear tabla si no existe
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cards (
            oracle_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            flavor_name TEXT,
            cmc REAL,
            type_line TEXT,
            oracle_text TEXT,
            colors TEXT,
            keywords TEXT,
            mana_cost TEXT,
            game_changer INTEGER DEFAULT 0
        )
    """)

    # Índices para búsquedas eficientes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cards_name ON cards(name);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cards_flavor ON cards(flavor_name);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cards_game_changer ON cards(game_changer);")

    # 2. Cargar en memoria los oracle_id que YA EXISTEN en la base de datos
    cursor.execute("SELECT oracle_id FROM cards")
    existentes_en_db = {row[0] for row in cursor.fetchall()}
    print(f"ℹ️ Registros ya guardados en la BD: {len(existentes_en_db)}")

    cards_to_insert = {}
    line_count = 0

    print(f"📖 Procesando el archivo JSONL línea por línea ({JSON_BULK_FILE})...")
    
    with open(JSON_BULK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            card = json.loads(line)  # Parsea la línea actual

            # Filtrar cartas no jugables
            if card.get("layout") in ["token", "art_series", "double_faced_token"]:
                continue

            oracle_id = card.get("oracle_id")
            if not oracle_id:
                continue

            # SI YA EXISTE EN LA BD, SE OMITE DE UNA
            if oracle_id in existentes_en_db:
                continue

            name = card.get("name")
            flavor_name = card.get("flavor_name", "")
            is_game_changer = 1 if card.get("game_changer") is True else 0

            # Manejo de múltiples versiones dentro del mismo archivo si aún no se ha metido a la DB
            if oracle_id in cards_to_insert:
                registro_previo = list(cards_to_insert[oracle_id])
                modificado = False
                
                if flavor_name and not registro_previo[2]:
                    registro_previo[2] = flavor_name
                    modificado = True

                if is_game_changer == 1 and registro_previo[9] == 0:
                    registro_previo[9] = 1
                    modificado = True

                if modificado:
                    cards_to_insert[oracle_id] = tuple(registro_previo)
                continue

            cmc = card.get("cmc", 0.0)
            type_line = card.get("type_line", "")
            keywords = ",".join(card.get("keywords", []))

            # Manejo de cartas multifaz (MDFCs, Transform, Split, Adventure)
            if "card_faces" in card and card["layout"] in ["transform", "modal_dfc", "split", "flip", "adventure"]:
                faces = card["card_faces"]
                oracle_text = " // ".join([f.get("oracle_text", "") for f in faces if f.get("oracle_text")])
                mana_cost = " // ".join([f.get("mana_cost", "") for f in faces if f.get("mana_cost")])
                
                colors_set = set()
                for f in faces:
                    colors_set.update(f.get("colors", []))
                colors = ",".join(colors_set)
            else:
                oracle_text = card.get("oracle_text", "")
                mana_cost = card.get("mana_cost", "")
                colors = ",".join(card.get("colors", []))

            cards_to_insert[oracle_id] = (
                oracle_id, name, flavor_name, cmc, type_line, oracle_text, colors, keywords, mana_cost, is_game_changer
            )

    records = list(cards_to_insert.values())
    print(f"📊 Líneas leídas: {line_count}. Novedades a insertar: {len(records)}.")

    if records:
        print(f"⚡ Insertando {len(records)} cartas nuevas en {DB_FILE}...")
        # Usamos INSERT OR IGNORE para asegurar cero duplicados a nivel BD
        cursor.executemany("""
            INSERT OR IGNORE INTO cards 
            (oracle_id, name, flavor_name, cmc, type_line, oracle_text, colors, keywords, mana_cost, game_changer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, records)
        conn.commit()
    else:
        print("✨ No hay cartas nuevas por agregar, la base de datos ya está al día.")

    conn.close()
    print("🎉 ¡Listo, pa! Tu base de datos quedó procesada impecablemente.")

if __name__ == "__main__":
    poblar_db_desde_jsonl_local()