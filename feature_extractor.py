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
        # Excluir autodaño / painlands (ej. Llanowar Wastes, Mana Confluence)
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
            r"rather than pay (this spell's|its) mana cost",
            r"put (a|an|target|any) .* (creature|permanent|artifact) .* onto the battlefield",
            r"return target .* from your graveyard to the battlefield"
        ]
        return 1.0 if any(re.search(p, oracle_text) for p in patterns) else 0.0

    @staticmethod
    def _check_fast_mana_or_ritual(oracle_text: str, card_name: str, raw_cmc: float) -> float:
        # Fast Mana Icónicos o Rituals de 1 uso
        fast_mana_names = [
            "sol ring", "mana crypt", "jeweled lotus", "lotus petal", 
            "chrome mox", "mox diamond", "mox opal", "mana vault", 
            "grim monolith", "dark ritual", "culling the weak", "cabal ritual"
        ]
        if card_name.lower() in fast_mana_names:
            return 1.0
        
        # Rituals que dan más maná del que cuestan en el mismo turno
        if "add {" in oracle_text and raw_cmc <= 1.0 and "instant" in oracle_text:
            return 1.0
            
        return 0.0

    @staticmethod
    def _check_ramp(oracle_text: str, type_line: str, is_fast_mana: float) -> float:
        # Si es fast mana, no lo contamos como Ramp estándar sostenido
        if is_fast_mana > 0.0:
            return 0.0

        # Si las tierras no buscan ni ponen tierras extra, no son ramp
        if "land" in type_line and not any(k in oracle_text for k in ["put", "search", "onto the battlefield"]):
            return 0.0

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

    # --- MÉTODO PRINCIPAL DE EXTRACCIÓN (28D) ---
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
            return np.zeros(28, dtype=np.float32)

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

        # 4. Mecánicas (14D)
        is_mld = self._check_mass_land_denial(oracle_text)
        is_board_wipe = 1.0 if (
            ("destroy all" in oracle_text or "exile all" in oracle_text) and not is_mld
        ) else 0.0

        # Lógica de Tutores Dividida por CMC
        is_tutor = ("search your library" in oracle_text and "land" not in oracle_text)
        is_fast_tutor = 1.0 if (is_tutor and raw_cmc <= 2.0) else 0.0
        is_slow_tutor = 1.0 if (is_tutor and raw_cmc > 2.0) else 0.0

        # Lógica de Fast Mana vs Ramp Normal
        is_fast_mana = self._check_fast_mana_or_ritual(oracle_text, card_name, raw_cmc)
        is_normal_ramp = self._check_ramp(oracle_text, type_line, is_fast_mana)

        vec_mechanics = [
            is_normal_ramp,                                         # 1. Ramp Tradicional
            is_fast_mana,                                           # 2. Fast Mana / Rituals (NUEVO)
            self._check_card_draw(oracle_text),                     # 3. Draw
            is_fast_tutor,                                          # 4. Fast Tutor CMC<=2 (NUEVO)
            is_slow_tutor,                                          # 5. Slow Tutor CMC>=3 (NUEVO)
            self._check_removal(oracle_text),                       # 6. Removal
            1.0 if ("counter target spell" in oracle_text) else 0.0,# 7. Counter
            is_board_wipe,                                         # 8. Board Wipe
            1.0 if ("extra turn" in oracle_text) else 0.0,          # 9. Extra turns
            is_mld,                                                # 10. MLD
            self._check_spell_copy(oracle_text),                    # 11. Copy/Storm
            self._check_cheat_mana(oracle_text),                   # 12. Cheat Mana / Reanimate
            self._check_burn_ping(oracle_text),                    # 13. Burn / Drain / Ping
            self._check_trigger_multiplication(oracle_text)        # 14. Trigger Multipliers / Extra Attacks
        ]

        # 5. Game Changer (1D)
        vec_game_changer = [float(game_changer or 0)]

        # Vector Final: 1 + 5 + 7 + 14 + 1 = 28 Dimensiones
        return np.array(
            vec_cmc + vec_colors + vec_types + vec_mechanics + vec_game_changer, 
            dtype=np.float32
        )