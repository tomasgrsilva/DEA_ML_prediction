import pandas as pd
from deadb.common.database import engine


def get_molecules():
    with engine.connect() as conn:
        return pd.read_sql("""
            SELECT id, name, molecular_weight, atomic_composition, halogen_positions, dipole_moment
            FROM molecules
        """, conn)


def get_results():
    with engine.connect() as conn:
        return pd.read_sql("""
            SELECT id, paper_molecule_id, peak, fragment_id
            FROM results
        """, conn)

def get_paper_molecule_links():
    with engine.connect() as conn:
        return pd.read_sql("""
            SELECT id, molecule_id
            FROM paper_molecule_link
        """, conn)

def get_bde():
    with engine.connect() as conn:
        return pd.read_sql("""
            SELECT id, molecule_id, fragment_id, bde_values
            FROM bdes
        """, conn)

def get_fragment():
    with engine.connect() as conn:
        return pd.read_sql("""
            SELECT id, electron_affinity
            FROM fragments
        """, conn)