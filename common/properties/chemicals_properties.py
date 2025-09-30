from chemicals.elements import molecular_weight
from chemicals.dipole import dipole_moment
import periodictable
from rdkit import Chem
from collections import defaultdict
import re
from common.properties.atoms import halogen_symbols

KCAL_TO_EV = 0.0433641

def formula_to_composition(formula: str) -> dict:
    """Convert a molecular formula string into an element composition dictionary."""
    parsed_formula = periodictable.formula(formula)  # Parse the formula using periodictable
    return {el.symbol: count for el, count in parsed_formula.atoms.items()}

def MW_from_formula(formula: str) -> float:
    """
    Calculate the molecular weight using the formula string.
    """
    # Convert formula to atomic composition
    composition = formula_to_composition(formula)

    # Calculate molecular weight using chemicals
    mw = molecular_weight(composition)
    return mw


def extract_halogen_positions(smiles):
    """Finds positions of halogens in a molecule from its SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None  # Return None if the SMILES string is invalid

    halogen_positions = defaultdict(list)

    for atom in mol.GetAtoms():
        if atom.GetSymbol() in halogen_symbols:  # Check for halogen atoms
            halogen_positions[atom.GetSymbol()].append(atom.GetIdx() + 1)  # Convert to 1-based index

    return dict(halogen_positions)

def extract_atomic_composition(formula):
    """Parses a molecular formula (e.g., C2H5Cl) into a dictionary {element: count}."""
    if not isinstance(formula, str) or not formula:
        return None  # Return None if input is invalid

    pattern = r'([A-Z][a-z]?)(\d*)'  # Matches element symbols and their counts
    composition = defaultdict(int)

    matches = re.findall(pattern, formula)
    for element, count in matches:
        composition[element] += int(count) if count else 1  # Default to 1 if count is missing

    return dict(composition)

def dip_moment(casrn):
    return dipole_moment(casrn)
