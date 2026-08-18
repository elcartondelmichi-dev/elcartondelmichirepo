import sqlite3
import numpy as np
import os
import re

DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(DIRECTORIO_ACTUAL, "prueba.db")

class CardFeatureExtractor:
    def __init__(self, db_path=DB_FILE):
        self.db_path = db_path

    def _obtener_conexion(self):
        return sqlite3.connect(self.db_path)

    # --- HELPERS REGEX OPTIMIZADOS ---
    @staticmethod
    def _check_mass_land_denial(oracle_text: str) -> float:
        patterns = [
            r"destroy all lands",
            r"exile all lands",
            r"lands don't untap",
            r"lands lose all land types",
            r"destroy all permanents",
            r"exile all permanents"
        ]
        return 1.0 if any(re.search(p, oracle_text) for p in patterns) else 0.0

    @staticmethod
    def _check_spell_copy(oracle_text: str) -> float:
        patterns = [
            r"copy target (instant|sorcery|spell)",
            r"copy next (instant|sorcery|spell)",
            r"copy that spell",
            r"copy it",
            r"copies of",
            r"\bstorm\b",
            r"\breplicate\b",
            r"\bconspire\b"
        ]
        return 1.0 if any(re.search(p, oracle_text) for p in patterns) else 0.0

    @staticmethod
    def _check_card_draw(oracle_text: str) -> float:
        pattern = r"\bdraw(s)?\b\s+([^.\n]*?\b)?card(s)?\b"
        return 1.0 if re.search(pattern, oracle_text) else 0.0

    @staticmethod
    def _check_removal(oracle_text: str) -> float:
        # Excluir explícitamente autodaño / painlands (ej. Llanowar Wastes, Mana Confluence)
        if "deals 1 damage to you" in oracle_text or "deals 2 damage to you" in oracle_text:
            return 0.0

        patterns = [
            r"destroy target",
            r"exile target",
            r"target creature gets",
            r"deals? \d+ damage to target",
            r"deals? x damage to target",
            r"target player sacrifices"
        ]
        return 1.0 if any(re.search(p, oracle_text) for p in patterns) else 0.0

    @staticmethod
    def _check_cheat_mana(oracle_text: str) -> float:
        patterns = [
            # Alternativas de casteo libre / condicional
            r"rather than pay (this spell's|its) mana cost",
            # Atrapa Kaalia, Sneak Attack, Quicksilver Amulet, etc.
            r"put (a|an|target|any) .* (creature|permanent|artifact) .* onto the battlefield",
            # Reanimadores
            r"return target .* from your graveyard to the battlefield"
        ]
        return 1.0 if any(re.search(p, oracle_text) for p in patterns) else 0.0

    @staticmethod
    def _check_ramp(oracle_text: str) -> float:
        # Si la carta te obliga a poner la criatura en el fondo de la biblioteca (ej. Unlucky Cabbage Merchant),
        # suele ser un tutor / fix muy ineficiente y no ramp acelerador sostenido.
        if "bottom of its owner's library" in oracle_text:
            return 0.0

        patterns = [
            r"add\s+\{[wubrgc0-9x]\}\b",
            r"add\s+mana",
            r"search your library for a (basic )?land card and put (it|them) onto the battlefield"
        ]
        return 1.0 if any(re.search(p, oracle_text) for p in patterns) else 0.0

    @staticmethod
    def _check_burn_ping(oracle_text: str) -> float:
        # Excluir daño a ti mismo
        if "deals 1 damage to you" in oracle_text or "deals 2 damage to you" in oracle_text:
            return 0.0

        patterns = [
            r"deals? \d+ damage to (each|target|opponent)",
            r"whenever .* enters.* deals",
            r"whenever .* dies.* deals",
            r"each opponent loses \d+ life",
            r"whenever .* attacks.* deals"
        ]
        return 1.0 if any(re.search(p, oracle_text) for p in patterns) else 0.0

    @staticmethod
    def _check_trigger_multiplication(oracle_text: str) -> float:
        patterns = [
            r"triggers an additional time",
            r"trigger an additional time",
            r"triggers? twice",
            r"additional attack phase",
            r"if a ability .* triggers"
        ]
        return 1.0 if any(re.search(p, oracle_text) for p in patterns) else 0.0

    # --- MÉTODO PRINCIPAL DE EXTRACCIÓN ---
    def extract_features(self, card_name: str) -> np.ndarray:
        conn = self._obtener_conexion()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT cmc, colors, type_line, oracle_text, game_changer 
            FROM cards 
            WHERE name = ? 
               OR flavor_name = ? 
               OR name LIKE ? || ' // %'
            LIMIT 1
        """, (card_name, card_name, card_name))
        
        row = cursor.fetchone()
        conn.close()

        if not row:
            return np.zeros(26, dtype=np.float32)

        cmc, colors_str, type_line, oracle_text, game_changer = row
        
        type_line = (type_line or "").lower()
        oracle_text = (oracle_text or "").lower()
        colors = colors_str.split(",") if colors_str else []

        # 1. Coste Convertido Normalizado (1D)
        raw_cmc = float(cmc or 0.0)
        vec_cmc = [min(raw_cmc / 10.0, 1.0)]

        # 2. Colores WUBRG (5D)
        vec_colors = [
            1.0 if 'W' in colors else 0.0,
            1.0 if 'U' in colors else 0.0,
            1.0 if 'B' in colors else 0.0,
            1.0 if 'R' in colors else 0.0,
            1.0 if 'G' in colors else 0.0,
        ]

        # 3. Tipos de Carta (7D)
        vec_types = [
            1.0 if 'creature' in type_line else 0.0,
            1.0 if 'land' in type_line else 0.0,
            1.0 if 'artifact' in type_line else 0.0,
            1.0 if 'enchantment' in type_line else 0.0,
            1.0 if 'instant' in type_line else 0.0,
            1.0 if 'sorcery' in type_line else 0.0,
            1.0 if 'planeswalker' in type_line else 0.0,
        ]

        # 4. Mecánicas (12D)
        is_mld = self._check_mass_land_denial(oracle_text)
        is_board_wipe = 1.0 if (
            ("destroy all" in oracle_text or "exile all" in oracle_text) and not is_mld
        ) else 0.0

        vec_mechanics = [
            self._check_ramp(oracle_text),
            self._check_card_draw(oracle_text),
            1.0 if ("search your library" in oracle_text and "land" not in oracle_text) else 0.0,
            self._check_removal(oracle_text),
            1.0 if ("counter target spell" in oracle_text) else 0.0,
            is_board_wipe,
            1.0 if ("extra turn" in oracle_text) else 0.0,
            is_mld,
            self._check_spell_copy(oracle_text),
            self._check_cheat_mana(oracle_text),
            self._check_burn_ping(oracle_text),
            self._check_trigger_multiplication(oracle_text)
        ]

        # 5. Game Changer (1D)
        vec_game_changer = [float(game_changer or 0)]

        return np.array(
            vec_cmc + vec_colors + vec_types + vec_mechanics + vec_game_changer, 
            dtype=np.float32
        )