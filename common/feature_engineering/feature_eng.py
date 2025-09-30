from common.properties.atoms import halogen_symbols
import pandas as pd
import numpy as np

def expand_atomic_composition(df_molecules):
    """
    Expands the 'atomic_composition' column of df_molecules into separate num_X columns,
    and returns the updated DataFrame.
    """

    def expand_dict(composition):
        """Helper to expand a dictionary like {'C': 3, 'H': 8} into {'num_C': 3, 'num_H': 8}."""
        if not isinstance(composition, dict):
            return {}
        return {f'num_{element}': count for element, count in composition.items()}

    # Apply the helper to each row and create a new DataFrame
    expanded_features = df_molecules['atomic_composition'].apply(expand_dict)
    df_atomic_features = pd.DataFrame(expanded_features.tolist(), index=df_molecules.index)

    # Combine original DataFrame with new atomic composition columns
    df_with_composition = pd.concat([df_molecules, df_atomic_features], axis=1)

    return df_with_composition


def expand_halogen_positions(df_molecules):
    """
    Expands halogen positions into separate columns dynamically.

    Args:
        df_molecules: df with info from molecules table, including 'halogen_positions'

    Returns:
        df_molecules: updated dataframe
    """
    max_positions_dict = {halogen: 0 for halogen in halogen_symbols}

    # Find the maximum number of positions for each halogen
    for positions in df_molecules['halogen_positions']:
        if isinstance(positions, dict):  # Ensure it's a dictionary
            for halogen in halogen_symbols:
                if halogen in positions:
                    max_positions_dict[halogen] = max(max_positions_dict[halogen], len(positions[halogen]))

    # Create new columns for each halogen's positions
    for halogen, max_positions in max_positions_dict.items():
        for i in range(1, max_positions + 1):
            column_name = f'{halogen}{i}'
            df_molecules[column_name] = df_molecules.apply(
                lambda row: row['halogen_positions'].get(halogen, [None] * max_positions)[i - 1]
                if isinstance(row['halogen_positions'], dict) and len(row['halogen_positions'].get(halogen, [])) >= i
                else None, axis=1
            )

    return df_molecules

def expand_bde_values(df_bde):
    """Creates BDE features through JSON field bde_values"""

    # Compute features
    df_bde['lowest_bde'] = df_bde['bde_values'].apply(min)
    df_bde['mean_bde'] = df_bde['bde_values'].apply(lambda x: sum(x) / len(x))

    return df_bde


def separate_bde_values(df_bde):
    """
    Creates three columns from the bde_values JSON field, assigning each value to a separate column.
    If fewer than three values exist, remaining columns are filled with 0.

    Args:
        df_bde: DataFrame containing a 'bde_values' column with JSON lists of BDE values.

    Returns:
        DataFrame with new columns 'bde_value_1', 'bde_value_2', 'bde_value_3'.
    """
    # Initialize the new columns with zeros
    df_bde['bde_value_1'] = 0.0
    df_bde['bde_value_2'] = 0.0
    df_bde['bde_value_3'] = 0.0

    # Function to assign BDE values to columns
    def assign_bde_values(bde_list):
        # Handle empty or invalid lists
        if not isinstance(bde_list, list):
            return [0.0, 0.0, 0.0]

        # Pad or truncate to exactly three values
        padded = bde_list[:3] + [0.0] * (3 - len(bde_list[:3]))
        return padded

    # Apply the function and assign to columns
    bde_expanded = df_bde['bde_values'].apply(assign_bde_values)
    df_bde[['bde_value_1', 'bde_value_2', 'bde_value_3']] = pd.DataFrame(
        bde_expanded.tolist(),
        index=df_bde.index
    )

    return df_bde


def create_energy_range_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a categorical target column to the dataframe based on peak energy ranges.

    Args:
        df: DataFrame containing a 'peak' column with energy values in eV.

    Returns:
        DataFrame with an additional 'energy_range' column containing one of:
        '0-1eV', '1-2eV', '2-3eV', '3eV+'.
    """

    def assign_range(peak):
        if 0 <= peak <= 1:
            return "0-1eV"
        elif 1 < peak <= 2:
            return "1-2eV"
        elif peak > 2:
            return "2eV+"
        """
        elif 2 < peak <= 3:
            return "2-3eV"
        elif peak > 3:
            return "3eV+"
        """
        return np.nan  # Handle invalid peaks

    df['energy_range'] = df['peak'].apply(assign_range)
    return df

