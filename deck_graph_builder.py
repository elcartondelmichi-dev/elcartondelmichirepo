import os
import sqlite3
import torch
import numpy as np
from feature_extractor import CardFeatureExtractor, DB_FILE
from deck_parser import parse_moxfield_deck

class DeckGraphBuilder:
    def __init__(self, db_path=DB_FILE):
        self.db_path = db_path
        self.extractor = CardFeatureExtractor(db_path=db_path)

    def _get_card_mechanics(self, card_name: str):
        """Consulta robusta a SQLite para resolver MDFCs, carátulas y flavor names."""
        clean_name = card_name.split("//")[0].strip() if "//" in card_name else card_name.strip()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 1. Busca por coincidencia exacta de nombre o flavor_name
        # 2. Si es MDFC, busca por 'Sink into Stupor %' para atrapar la barra doble '//'
        cursor.execute("""
            SELECT cmc, type_line, oracle_text, game_changer 
            FROM cards 
            WHERE name = ? 
               OR flavor_name = ? 
               OR name LIKE ? || ' // %'
            LIMIT 1
        """, (clean_name, clean_name, clean_name))
        
        row = cursor.fetchone()
        conn.close()

        if row:
            cmc, type_line, oracle_text, game_changer = row
            return {
                "name": clean_name,
                "cmc": cmc or 0,
                "type": (type_line or "").lower(),
                "text": (oracle_text or "").lower(),
                "game_changer": game_changer or 0
            }
        
        return {"name": clean_name, "cmc": 0, "type": "", "text": "", "game_changer": 0}

    def build_graph_from_decklist(self, decklist_names: list):
            """
            decklist_names: Lista de nombres de las cartas del mazo (100 en total).
            Devuelve:
            - x: Tensor de características de nodos [N, 28]
            - edge_index: Tensor de conexiones dinámicas raras/esparsas [2, num_edges]
            """
            features_list = []
            cards_mechanics = []

            # 1. Obtener la dimensión real que devuelve el extractor (28D)
            sample_vec = self.extractor.extract_features("Sol Ring")
            EXPECTED_DIM = len(sample_vec) if sample_vec is not None else 28

            for card_name in decklist_names:
                clean_name = card_name.split("//")[0].strip() if "//" in card_name else card_name.strip()
                
                # Extraer vector de características (28D)
                vec = self.extractor.extract_features(clean_name)
                
                if vec is not None and len(vec) == EXPECTED_DIM:
                    features_list.append(vec)
                else:
                    features_list.append(np.zeros(EXPECTED_DIM, dtype=np.float32))
                
                # Metadatos para evaluar sinergia
                mech = self._get_card_mechanics(clean_name)
                cards_mechanics.append(mech)

            num_cards = len(cards_mechanics)
            if num_cards == 0:
                return torch.empty((0, EXPECTED_DIM), dtype=torch.float), torch.empty((2, 0), dtype=torch.long)

            x = torch.tensor(np.array(features_list), dtype=torch.float)
            edges = set()

            def add_edge(i, j):
                if i != j:
                    edges.add((i, j))
                    edges.add((j, i))

            # 2. Generación Rala de Aristas (Sparse Synergy Graph)
            for i in range(num_cards):
                c1 = cards_mechanics[i]
                t1 = c1.get("text", "")
                type1 = c1.get("type", "")

                for j in range(i + 1, num_cards):
                    c2 = cards_mechanics[j]
                    t2 = c2.get("text", "")
                    type2 = c2.get("type", "")

                    # --- REGLA 1: UNTAP / MOTORES ---
                    c1_untaps = "untap" in t1
                    c2_untaps = "untap" in t2
                    c1_engine = ("add " in t1 or "draw" in t1) and c1["cmc"] <= 3
                    c2_engine = ("add " in t2 or "draw" in t2) and c2["cmc"] <= 3

                    if (c1_untaps and c2_engine) or (c2_untaps and c1_engine):
                        add_edge(i, j)

                    # --- REGLA 2: TUTORES DIRECTOS (Búsqueda real en biblioteca) ---
                    c1_tutor = "search your library" in t1
                    c2_tutor = "search your library" in t2

                    if (c1_tutor and c2["cmc"] <= 2 and "land" not in type2) or \
                    (c2_tutor and c1["cmc"] <= 2 and "land" not in type1):
                        add_edge(i, j)

                    # --- REGLA 3: RED DE GAME CHANGERS ---
                    if (c1["game_changer"] and c2["game_changer"]) or \
                    (c1["game_changer"] and c2_tutor) or \
                    (c2["game_changer"] and c1_tutor):
                        add_edge(i, j)

                    # --- REGLA 4: FAST MANA ALIMENTANDO AMENAZAS / MOTORES GRANDES ---
                    c1_fast = ("add " in t1 or "mana" in t1) and c1["cmc"] <= 1
                    c2_fast = ("add " in t2 or "mana" in t2) and c2["cmc"] <= 1

                    if (c1_fast and c2["cmc"] >= 5) or (c2_fast and c1["cmc"] >= 5):
                        add_edge(i, j)

            # 3. Empaquetar aristas en Tensor PyTorch Geometric
            if len(edges) > 0:
                edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            return x, edge_index

    
# --- PRUEBA CON TU MAZO REAL DE 100 CARTAS ---
if __name__ == "__main__":
    builder = DeckGraphBuilder()

    # Mazo real de Lathril Golgari (100 cartas completas)
    mazo_lathril_text = """
    1 Aerith's Curaga Magic
    1 Akroma's Memorial
    1 Assassin's Trophy
    1 Beast Whisperer
    1 Beast Within
    1 Beseech the Mirror
    1 Bloodline Bidding
    1 Canopy Tactician
    1 Casualties of War
    1 Command Tower
    1 Cover of Darkness
    1 Craterhoof Behemoth
    1 Deathcap Glade
    1 Deathrite Shaman
    1 Defense of the Heart
    1 Diabolic Tutor
    1 Dwynen, Gilt-Leaf Daen
    1 Eclipsed Realms
    1 Elderfang Venom
    1 Elven Ambush
    1 Elvish Archdruid
    1 Elvish Champion
    1 Elvish Guidance
    1 Elvish Harbinger
    1 Elvish Mystic
    1 Elvish Promenade
    1 Elvish Warmaster
    1 Exotic Orchard
    1 Ezuri, Renegade Leader
    1 Farhaven Elf
    1 Fog
    1 Forest
    1 Forest
    1 Forest
    1 Forest
    1 Forest
    1 Forest
    1 Forest
    1 Forest
    2 Forest
    1 Fyndhorn Elves
    1 Galadhrim Ambush
    1 Genesis Wave
    1 Gilt-Leaf Palace
    1 Gloom Ripper
    1 Growing Rites of Itlimoc // Itlimoc, Cradle of the Sun
    1 Harald, King of Skemfar
    1 Heritage Druid
    1 Immaculate Magistrate
    1 Imperious Perfect
    1 Invasion of Ikoria // Zilortha, Apex of Ikoria
    1 Joraga Warcaller
    1 Leaf-Crowned Visionary
    1 Llanowar Elves
    1 Llanowar Wastes
    1 Lys Alana Huntmaster
    1 Marwyn, the Nurturer
    1 Morcant's Loyalist
    1 Nissa Revane
    1 Overwhelming Stampede
    1 Pact of the Serpent
    1 Path of Ancestry
    1 Poison-Tip Archer
    1 Priest of Titania
    1 Quirion Ranger
    1 Reclamation Sage
    1 Ruthless Winnower
    1 Shaman of the Pack
    1 Skemfar Shadowsage
    1 Sol Ring
    1 Staff of Domination
    1 Swamp
    1 Swamp
    1 Swamp
    1 Swamp
    1 Swamp
    1 Swamp
    1 Swamp
    1 Swamp
    1 Swamp
    1 Swamp
    1 Swamp
    1 Swamp
    1 Taunting Elf
    1 Throne of the God-Pharaoh
    1 Timberwatch Elf
    1 Tyvar Kell
    1 Tyvar the Bellicose
    1 Tyvar, Jubilant Brawler
    1 Virulent Emissary
    1 Wellwisher
    1 Wirewood Channeler
    1 Wirewood Lodge
    1 Wirewood Pride
    1 Wirewood Symbiote
    1 Wolverine Riders
    1 Woodland Cemetery
    1 Yavimaya, Cradle of Growth
    1 Lathril, Blade of the Elves
    """

    # 1. Parsear el mazo a lista plana de nombres
    raw_cards = parse_moxfield_deck(mazo_lathril_text)
    
    # 2. Construir el grafo
    x, edge_index = builder.build_graph_from_decklist(raw_cards)

    # 3. Validaciones matemáticas del Grafo
    expected_edges = 100 * 99  # 9,900 aristas para un grafo completo de 100 nodos sin auto-loops

    print("\n" + "="*50)
    print("📊 RESULTADO DE LA PRUEBA DEL GRAFO DE 100 CARTAS")
    print("="*50)
    print(f"🃏 Cartas procesadas por el parser : {len(raw_cards)}")
    print(f"📐 Forma del Tensor de Nodos (X)   : {x.shape} (Esperado: [100, 23])")
    print(f"🔗 Forma de Edge Index             : {edge_index.shape} (Esperado: [2, {expected_edges}])")
    print("="*50)

    if x.shape == (100, 28) and edge_index.shape[1] == expected_edges:
        print("✅ ¡Éxito! El grafo se generó correctamente para la GNN.")
    else:
        print("❌ Hubo una discrepancia en las dimensiones del grafo.")