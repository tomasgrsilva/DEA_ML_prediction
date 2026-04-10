import pandas as pd
from deadb.common.database import engine
from sqlalchemy import text

def get_table(table_name: str):
    """
    Generic function to fetch all columns from a given table.

    Args:
        table_name (str): Name of the table to fetch.

    Returns:
        pd.DataFrame: DataFrame containing all data from the table.
    """
    query=f"SELECT * FROM {table_name}"
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def get_molecules():
    """Fetch all data from the molecules table."""
    return get_table("molecules")
def tabela_resultados():
    """Fetch all data from the results table."""
    return get_table("results")

def get_results():
    """Fetch all data from the results table."""
    return get_table("results")


def get_paper_molecule_links():
    """Fetch all data from the paper_molecule_link table."""
    return get_table("paper_molecule_link")


def get_families():
    """Fetch all data from the families table."""
    return get_table("families")


def get_setups():
    """Fetch all data from the setups table."""
    return get_table("setups")


def get_papers():
    """Fetch all data from the papers table."""
    return get_table("papers")


def get_fragments():
    """Fetch all data from the fragments table."""
    return get_table("fragments")
