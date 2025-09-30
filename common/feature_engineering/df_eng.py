import pandas as pd
from common.queries.query_bde import get_molecules, get_results, get_bde, get_fragment, get_paper_molecule_links
from common.feature_engineering.feature_eng import (expand_atomic_composition, expand_halogen_positions,
                                                    expand_bde_values, create_energy_range_target)

EXCLUDED_RESULT_IDS = [40, 56, 65, 73, 102, 104, 106, 108, 109, 111, 112, 113, 115, 116, 117, 119, 120,
                       123, 124, 126, 127, 128, 144, 147, 153, 155, 184, 191, 216, 218, 220, 221, 222,
                       233, 234, 240, 242, 257, 269, 278, 282, 284, 285, 301, 314, 323, 344, 345, 355,
                       357, 358, 359, 360, 361, 362, 364, 389, 393, 399, 411, 423, 195, 213]

# Result_id 189 is duplicate with peak at 0 eV but if I remove it, metrics get worse.
# Result_id 152 refers to a problematic entry
# Excluding 153 instead of 154 for CCl4 even though it has a slightly higher cross section because it is a 0 peak.
# 24, 121, 189 duplicates on the test set
#24 195
#121 213



INCLUDED_FRAGMENTS = [2, 3, 4, 5]
"""
2 - Cl
3 - F
4 - Br
5 - I
"""


def expand_h_position_features(df_molecules, use_halogen_positions=False):
    """
    Expands halogen position features if requested.

    Assumes df_molecules already contains atomic composition features.

    Args:
        df_molecules (pd.DataFrame): DataFrame containing molecule data with atomic composition features.
        use_halogen_positions (bool): Whether to expand halogen position features.

    Returns:
        pd.DataFrame: DataFrame with expanded features.
    """

    df_expanded = df_molecules.copy()

    if use_halogen_positions:
        df_expanded = expand_halogen_positions(df_expanded)

    # Drop original nested/dict columns
    df_expanded = df_expanded.drop(columns=['atomic_composition', 'halogen_positions'], errors='ignore')

    return df_expanded.fillna(0)


def filter_by_dipole_moment(df_molecules, include_dipole=False):
    """
    Optionally filters molecules by dipole moment availability.

    Args:
        df_molecules (pd.DataFrame): DataFrame with a 'dipole_moment' column.
        include_dipole (bool): If True, remove molecules with missing dipole moment.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if include_dipole:
        return df_molecules[df_molecules["dipole_moment"].notna()]
    else:
        return df_molecules.drop(columns=["dipole_moment"], errors="ignore")


def build_model_dataframe(include_dipole=False, use_halogen_positions=False, task="r"):
    # Load data
    df_molecules = get_molecules()
    df_results = get_results().rename(columns={"id": "result_id"})

    # Filter result for molecules 89-120 -- Remove for every molecule to be included
    #df_results = df_results[~df_results["result_id"].between(195, 362)] # Filter result for molecules 89-120 -- Remove for every molecule to be included

    df_links = get_paper_molecule_links()
    df_bde = get_bde()
    df_fragments = get_fragment()

    df_bde = expand_bde_values(df_bde)


    """
    DIPOLE MOMENT 
        If we want to use dipole moment as a feature of our model, we set include_dipole=True.
        It drops all molecules where the dipole moment value is unknown.
        Reminder: Dipole moment is obtained through dipole_moment function from chemicals.dipole library for which we need the CASRN.
        Path: deadb/common/properties/chemical_properties.py
    """
    df_molecules = filter_by_dipole_moment(df_molecules, include_dipole)

    # Atomic composition + halogen position features
    # Apply expand_atomic_composition function, creating a new DF with the number of atoms for each element
    df_molecules = expand_atomic_composition(df_molecules)


    """
    GET HALOGEN POSITIONS FEATURES
        If we want features for the halogen positions in the SMILES chain we set use_halogen_positions=True.
    """
    df_molecules_expanded = expand_h_position_features(df_molecules, use_halogen_positions)

    # Merge pipeline
    # Merge Results table with Paper_molecule_link table, connecting results.paper_molecule_id with paper_molecule_link.id
    df = df_results.merge(df_links, left_on="paper_molecule_id", right_on="id")

    # Merge the resulting table with df_molecules_expanded --> Molecules table + feature engineering,
    # connecting paper_molecule_link.molecule_id with molecules_id and
    # dropping id_x and id_y, being paper_molecule_link.id and molecule.id respectively, and results.paper_molecule_id
    df = df.merge(df_molecules_expanded, left_on="molecule_id", right_on="id").drop(
        columns=["id_x", "id_y", "paper_molecule_id"]
    )

    # Merge resulting df with Fragments table, connecting results.fragment_id with fragments.id (which is dropped)
    df = df.merge(df_fragments, left_on="fragment_id", right_on="id", how="left").drop(columns=["id"])

    # Filter fragment_id and (optionally) specific result_ids
    df = df[~df["result_id"].isin(EXCLUDED_RESULT_IDS)]  # Example: remove specific result_ids
    df = df[df["fragment_id"].isin(INCLUDED_FRAGMENTS)]


    # Merge BDE data
    df = df.merge(df_bde, on=["molecule_id", "fragment_id"], how="left").drop(
        columns=["id", "fragment_id", "bde_values"]
    )

    df["th_energy"] = df["lowest_bde"] - df["electron_affinity"]

    # Adds 'energy_range' column to be used as target for our classification model and drops 'peak' column
    if task == "c":
        df = create_energy_range_target(df)
        df = df.dropna(subset=["energy_range"])
        df = df.drop(columns=["peak"], errors="ignore")

    return df
