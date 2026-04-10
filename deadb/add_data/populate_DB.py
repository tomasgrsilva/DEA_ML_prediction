import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from deadb.common.database import engine
from deadb.common.models import (Molecule, Setup, Paper, PaperMoleculeLink, Fragment, Results, Family, Bde,
                                 Level1Category, Level2Category, Level3Category, Level4Category, Journal, Author,
                                 Institution, paper_author_link, paper_institution_link)

import logging
from common.properties.chemicals_properties import (MW_from_formula, extract_halogen_positions,
                                                    extract_atomic_composition, dip_moment, KCAL_TO_EV)


# Configure logging
logging.basicConfig(filename="db_insert.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the Excel file
file_path = r"C:\Users\Diogo Chikhi\OneDrive\Ambiente de Trabalho\Faculdade\5ano\Tese\tese_coisas_do_Tomas\DEAdb_BDE.xlsx"
sheets = pd.read_excel(file_path, sheet_name=None)

# Create a database session
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

def upsert_record(model, data, unique_field="id"):
    """Handles insert/update logic and logs whether a record is new or updated."""
    existing_record = session.query(model).filter_by(**{unique_field: data[unique_field]}).first()
    if existing_record:
        session.merge(model(**data))
        logging.info(f"🔄 Updated {model.__name__}: {data}")
    else:
        session.add(model(**data))
        logging.info(f"✅ Inserted {model.__name__}: {data}")

try:

    # Upsert Level Categories
    for sheet, model in [
        ("level1_categories", Level1Category),
        ("level2_categories", Level2Category),
        ("level3_categories", Level3Category),
        ("level4_categories", Level4Category)
    ]:
        if sheet in sheets:
            df = sheets[sheet]
            for record in df.dropna(how="all").to_dict(orient="records"):
                upsert_record(model, record, unique_field="name")

    # Upsert Families
    if "families" in sheets:
        df = sheets["families"]

        level1_ids = {c.name: c.id for c in session.query(Level1Category.name, Level1Category.id)}
        level2_ids = {c.name: c.id for c in session.query(Level2Category.name, Level2Category.id)}
        level3_ids = {c.name: c.id for c in session.query(Level3Category.name, Level3Category.id)}
        level4_ids = {c.name: c.id for c in session.query(Level4Category.name, Level4Category.id)}

        for row in df.dropna(how="all").to_dict(orient="records"):
            level1_id = level1_ids.get(row["level1_name"])
            level2_id = level2_ids.get(row["level2_name"])
            level4_id = level4_ids.get(row["level4_name"]) if row.get("level4_name") else None

            if not (level1_id and level2_id):
                logging.warning(f"⚠️ Skipping family {row.get('id')}: missing Level1 or Level2")
                continue

            # Parse Level3 names (comma separated in Excel)
            level3_list = []
            level3_names = row.get("level3_name")

            if level3_names and not pd.isna(level3_names):
                for name in level3_names.split(","):
                    name = name.strip()
                    if name in level3_ids:
                        level3_list.append(level3_ids[name])
                    else:
                        logging.warning(f"⚠️ Level3Category '{name}' not found for family {row.get('id')}")

            if not level3_list:
                logging.warning(f"⚠️ Skipping family {row.get('id')}: no valid Level3 categories")
                continue

            family = session.get(Family, row["id"])

            if not family:
                family = Family(
                    id=row["id"],
                    level1_id=level1_id,
                    level2_id=level2_id,
                    level4_id=level4_id
                )
                session.add(family)
                logging.info(f"🆕 Created Family {row['id']}")
            else:
                family.level1_id = level1_id
                family.level2_id = level2_id
                family.level4_id = level4_id
                logging.info(f"🔄 Updating Family {row['id']}")

            # Assign MANY-TO-MANY Level3 relationship
            family.level3_categories = session.query(Level3Category).filter(
                Level3Category.id.in_(level3_list)
            ).all()

            logging.info(f" Linked Level3 IDs {level3_list}")

    # Upsert Molecules, Families, and FamilyLevel3Link
    if "molecules" in sheets:
        df = sheets["molecules"]
        level1_ids = {c.name: c.id for c in session.query(Level1Category.name, Level1Category.id).all()}
        level2_ids = {c.name: c.id for c in session.query(Level2Category.name, Level2Category.id).all()}
        level3_ids = {c.name: c.id for c in session.query(Level3Category.name, Level3Category.id).all()}
        level4_ids = {c.name: c.id for c in session.query(Level4Category.name, Level4Category.id).all()}

        # Cache existing families by their level combinations
        family_cache = {}
        for f in session.query(Family).all():
            key = (f.level1_id, f.level2_id, frozenset([c.id for c in f.level3_categories]), f.level4_id)
            family_cache[key] = f.id

        for molecule in df.dropna(how="all").to_dict(orient="records"):
            formula = molecule.get("formula")
            smiles = molecule.get("smiles")
            casrn = molecule.get("casrn")

            # Calculate properties
            if formula:
                try:
                    molecule["molecular_weight"] = MW_from_formula(formula)
                    molecule["atomic_composition"] = extract_atomic_composition(formula)
                except Exception as e:
                    logging.warning(f"⚠️ Error calculating properties for {formula}: {e}")
                    molecule["molecular_weight"] = None
                    molecule["atomic_composition"] = None

            if smiles:
                try:
                    molecule["halogen_positions"] = extract_halogen_positions(smiles)
                except Exception as e:
                    logging.warning(f"⚠️ Error calculating halogen positions for {smiles}: {e}")
                    molecule["halogen_positions"] = None

            if "dipole_moment" not in molecule or pd.isna(molecule["dipole_moment"]):
                if casrn:
                    try:
                        molecule["dipole_moment"] = dip_moment(casrn)
                    except Exception as e:
                        logging.warning(f"⚠️ Error calculating dipole moment for {casrn}: {e}")
                        molecule["dipole_moment"] = None

            # Get level names from the molecule
            level1_name = molecule.get("level1_name")
            level2_name = molecule.get("level2_name")
            level3_name = molecule.get("level3_name")
            level4_name = molecule.get("level4_name")

            # Map names to existing IDs
            level1_id = level1_ids.get(level1_name)
            level2_id = level2_ids.get(level2_name)
            level4_id = level4_ids.get(level4_name) if level4_name else None

            if not all([level1_id, level2_id]):
                logging.warning(
                    f"⚠️ Skipping molecule {molecule.get('name')}: Invalid level name(s) - {level1_name}, {level2_name}")
                continue

            # Parse comma-separated level3_name values into a list of IDs
            level3_id_list = []
            if level3_name and not pd.isna(level3_name):
                for name in level3_name.split(","):
                    name = name.strip()
                    level3_id = level3_ids.get(name)
                    if level3_id:
                        level3_id_list.append(level3_id)
                    else:
                        logging.warning(f"⚠️ Level3Category {name} not found for molecule {molecule.get('name')}")

            if not level3_id_list:
                logging.warning(f"⚠️ No valid Level3Category IDs for molecule {molecule.get('name')}")
                continue

            # Check if this family combination exists
            family_key = (level1_id, level2_id, frozenset(level3_id_list), level4_id)
            if family_key not in family_cache:
                max_family_id = session.query(Family.id).order_by(Family.id.desc()).first()
                new_family_id = (max_family_id[0] + 1) if max_family_id else 1
                new_family = Family(
                    id=new_family_id,
                    level1_id=level1_id,
                    level2_id=level2_id,
                    level4_id=level4_id
                )

                new_family.level3_categories = session.query(Level3Category).filter(
                    Level3Category.id.in_(level3_id_list)
                ).all()

                session.add(new_family)
                logging.info(f"🆕 Created new family: {new_family_id} with Level3 IDs {level3_id_list}")
                family_cache[family_key] = new_family_id


            # Filter out level names before upserting molecule
            molecule_data = {k: v for k, v in molecule.items() if
                             k not in ["level1_name", "level2_name", "level3_name", "level4_name"]}
            molecule_data["family_id"] = family_cache[family_key]
            upsert_record(Molecule, molecule_data)
    if "setups" in sheets:
        df = sheets["setups"]
        for setup in df.dropna(how="all").to_dict(orient="records"):
            upsert_record(Setup, setup)
    # Upsert Papers
    if "papers" in sheets:
        df = sheets["papers"]
        setup_ids = {s.id for s in session.query(Setup.id).all()}
        journal_ids = {j.name: j.id for j in session.query(Journal.name, Journal.id).all()}
        author_ids = {a.name: a.id for a in session.query(Author.name, Author.id).all()}
        institution_ids = {i.name: i.id for i in session.query(Institution.name, Institution.id).all()}

        for paper in df.dropna(how="all").to_dict(orient="records"):
            # Handle journal
            journal_name = paper.get("journal_name")
            if journal_name and journal_name not in journal_ids:
                max_journal_id = session.query(Journal.id).order_by(Journal.id.desc()).first()
                new_journal_id = (max_journal_id[0] + 1) if max_journal_id else 1
                journal_data = {"id": new_journal_id, "name": journal_name}
                upsert_record(Journal, journal_data, unique_field="name")
                journal_ids[journal_name] = new_journal_id
                logging.info(f"🆕 Created new journal: {journal_data}")

            # Handle authors
            author_id_list = []
            author_names = paper.get("author_names")
            if author_names and not pd.isna(author_names):
                for name in author_names.split(","):
                    name = name.strip()
                    if name:
                        if name not in author_ids:
                            max_author_id = session.query(Author.id).order_by(Author.id.desc()).first()
                            new_author_id = (max_author_id[0] + 1) if max_author_id else 1
                            author_data = {"id": new_author_id, "name": name}
                            upsert_record(Author, author_data, unique_field="name")
                            author_ids[name] = new_author_id
                            logging.info(f"🆕 Created new author: {author_data}")
                        author_id_list.append(author_ids[name])

            # Handle institutions
            institution_id_list = []
            institution_names = paper.get("institution_names")
            if institution_names and not pd.isna(institution_names):
                for name in institution_names.split(";"):
                    name = name.strip()
                    if name:
                        if name not in institution_ids:
                            max_institution_id = session.query(Institution.id).order_by(
                                Institution.id.desc()).first()
                            new_institution_id = (max_institution_id[0] + 1) if max_institution_id else 1
                            institution_data = {"id": new_institution_id, "name": name}
                            upsert_record(Institution, institution_data, unique_field="name")
                            institution_ids[name] = new_institution_id
                            logging.info(f"🆕 Created new institution: {institution_data}")
                        institution_id_list.append(institution_ids[name])

            # Prepare paper data
            paper_data = {
                "id": paper["id"],
                "title": paper.get("title"),
                "publication_year": int(paper["publication_year"]) if pd.notna(
                    paper.get("publication_year")) else None,
                "doi": paper.get("doi"),
                "url": paper.get("url"),
                "journal_id": journal_ids.get(journal_name) if journal_name else None,
                "setup_id": paper.get("setup_id")
            }

            if paper_data["setup_id"] in setup_ids:
                # Upsert Paper
                upsert_record(Paper, paper_data)

                # Insert into paper_author_link
                paper_id = paper_data["id"]
                for author_id in author_id_list:
                    link_data = {"paper_id": paper_id, "author_id": author_id}
                    existing_link = session.query(paper_author_link).filter_by(
                        paper_id=paper_id, author_id=author_id
                    ).first()
                    if not existing_link:
                        session.execute(paper_author_link.insert().values(link_data))
                        logging.info(f"✅ Inserted paper_author_link: {link_data}")

                # Insert into paper_institution_link
                for institution_id in institution_id_list:
                    link_data = {"paper_id": paper_id, "institution_id": institution_id}
                    existing_link = session.query(paper_institution_link).filter_by(
                        paper_id=paper_id, institution_id=institution_id
                    ).first()
                    if not existing_link:
                        session.execute(paper_institution_link.insert().values(link_data))
                        logging.info(f"✅ Inserted paper_institution_link: {link_data}")
            else:
                logging.warning(f"⚠️ Skipping paper {paper['id']}: Invalid setup_id {paper_data['setup_id']}")


    # Upsert PaperMoleculeLink
    if "paper_molecule_link" in sheets:
        df = sheets["paper_molecule_link"]

        # Get all molecule names and their corresponding IDs
        molecule_ids = {m.name: m.id for m in session.query(Molecule.name, Molecule.id).all()}
        print(molecule_ids)

        # Get all paper IDs
        paper_ids = {p.id for p in session.query(Paper.id).all()}

        for link in df.dropna(how="all").to_dict(orient="records"):
            # If molecule_id is not given but molecule_name exists, get molecule_id
            molecule_name = link.get("molecule_name")  # Ensure molecule_name column exists in Excel
            if "molecule_id" not in link or pd.isna(link["molecule_id"]):
                if molecule_name in molecule_ids:
                    link["molecule_id"] = molecule_ids[molecule_name]
                    logging.info(f"🔄 Found molecule_id {link['molecule_id']} for molecule_name {molecule_name}")
                else:
                    logging.warning(f"⚠️ Molecule {molecule_name} not found in database!")

            # Keep only the fields needed for the database
            link = {
                "id": link["id"],
                "molecule_id": link["molecule_id"],
                "paper_id": link["paper_id"]
            }

            # Ensure that both molecule_id and paper_id are valid before inserting
            if link.get("molecule_id") in molecule_ids.values() and link.get("paper_id") in paper_ids:
                upsert_record(PaperMoleculeLink, link)

    # Upsert Fragments
    if "fragments" in sheets:
        df = sheets["fragments"]
        for fragment in df.dropna(how="all").to_dict(orient="records"):
            formula = fragment.get("formula")
            if formula:
                try:
                    fragment["molecular_weight"] = MW_from_formula(formula)
                except Exception as e:
                    logging.warning(f"⚠️ Error calculating molecular weight for fragment {formula}: {e}")
                    fragment["molecular_weight"] = None
            upsert_record(Fragment, fragment)

    # Upsert Results (Ensure valid paper_molecule_id & fragment_id)
    if "results" in sheets:
        df = sheets["results"]

        # Get all PaperMoleculeLink IDs
        paper_molecule_ids = {pml.id for pml in session.query(PaperMoleculeLink.id).all()}

        # Get all fragment formulas and their corresponding IDs

        #fragment_ids = {f.formula: f.id for f in session.query(Fragment.formula, Fragment.id).all()}
        ####

        fragment_ids = {}
        for f in session.query(Fragment).all():
            fragment_ids[(f.formula, f.electron_affinity)] = f.id
        ####
        for result in df.dropna(how="all").to_dict(orient="records"):
            # If fragment_id is not given but fragment_formula exists, get fragment_id
            ####
            fragment_formula = result.get("fragment_formula")
            ea = result.get("electron_affinity")
            ea = None if pd.isna(ea) else ea
            result.pop("electron_affinity", None)

            fragment_id = fragment_ids.get((fragment_formula, ea))
            if not fragment_id:
                matches = [fid for (f, e), fid in fragment_ids.items() if f == fragment_formula]
                if len(matches) == 1:
                    fragment_id = matches[0]
                else:
                    logging.warning(f"⚠️ Fragment formula {fragment_formula} not found in database!")
                    continue
            result["fragment_id"] = fragment_id
            ####
            #fragment_formula = result.get("fragment_formula")  # Ensure fragment_formula column exists in Excel
            #if "fragment_id" not in result or pd.isna(result["fragment_id"]):
             #   if fragment_formula in fragment_ids:
              #      result["fragment_id"] = fragment_ids[fragment_formula]
               #     logging.info(f"🔄 Found fragment_id {result['fragment_id']} for fragment_formula {fragment_formula}")
                #else:
                 #   logging.warning(f"⚠️ Fragment formula {fragment_formula} not found in database!")
                  #  continue  # Skip this result if no valid fragment_id is found
             ####
            if pd.isna(result.get("most_intense")):
                result["most_intense"] = None
            else:
                # Garante que converte corretamente para True/False
                result["most_intense"] = bool(result["most_intense"])
            # Drop fragment_formula since it's no longer needed
            result.pop("fragment_formula", None)
            result.pop("molecule_name", None)
            result.pop("molecule_formula", None)

            # Ensure that both paper_molecule_id and fragment_id are valid before inserting
            if result.get("paper_molecule_id") in paper_molecule_ids and result.get(
                    "fragment_id") in fragment_ids.values():
                upsert_record(Results, result)
            ##############################

    # Upsert BDEs
    if "bdes" in sheets:
        # Load the "bdes" sheet from the Excel file into a dataframe
        df = sheets["bdes"]

        # Query all molecule names and IDs in advance to optimize lookup
        # Create a dictionary mapping molecule names to their respective IDs
        molecule_ids = {m.name: m.id for m in session.query(Molecule.name, Molecule.id).all()}

        # Query all fragment formulas and IDs in advance to optimize lookup
        # Create a dictionary mapping fragment formulas to their respective IDs
        fragment_ids = {f.formula: f.id for f in session.query(Fragment.formula, Fragment.id).all()}

        # Iterate over each row (bde_entry) in the "bdes" dataframe
        for bde_entry in df.dropna(how="all").to_dict(orient="records"):
            # Get the molecule name from the current row
            molecule_name = bde_entry.get("molecule_name")

            # Get the fragment formula from the current row
            fragment_formula = bde_entry.get("fragment_formula")

            # Check if the molecule_name exists in the molecule_ids dictionary
            if "molecule_id" not in bde_entry or pd.isna(bde_entry["molecule_id"]):
                if molecule_name in molecule_ids:
                    # If it exists, assign the correct molecule_id to the current entry
                    bde_entry["molecule_id"] = molecule_ids[molecule_name]
                else:
                    # If molecule_name is not found, log a warning
                    logging.warning(f"⚠️ Molecule with name {molecule_name} not found!")

            # Check if the fragment_formula exists in the fragment_ids dictionary
            if fragment_formula in fragment_ids:
                # If it exists, assign the correct fragment_id to the current entry
                bde_entry["fragment_id"] = fragment_ids[fragment_formula]
            else:
                # If fragment_formula is not found, log a warning
                logging.warning(f"⚠️ Fragment with formula {fragment_formula} not found!")


            bde_values = []

            # Get bde_values. Get
            for key in ["bde_value_1", "bde_value_2", "bde_value_3"]:
                bde_value = bde_entry.get(key)
                if bde_value is not None and not pd.isna(bde_value):
                    try:
                        bde_values.append(round(float(bde_value) * KCAL_TO_EV, 4)) # Convert kcal/mol → eV
                    except ValueError:
                        logging.warning(f"⚠️ Invalid BDE value {bde_value} in {key} for {molecule_name}")

            # Ensure at least one valid BDE value exists
            if not bde_values:
                logging.warning(f"⚠️ No valid BDE values found for {molecule_name} - Skipping entry!")
                continue

            # Ensure bde_values is always a list
            bde_entry["bde_values"] = bde_values

            # Keep only the fields needed for the database
            bde_entry = {
                "id": bde_entry["id"],
                "molecule_id": bde_entry["molecule_id"],
                "fragment_id": bde_entry["fragment_id"],
                "bde_values": bde_entry["bde_values"],
            }

            # Ensure that both molecule_id and fragment_id are present in the current entry and are not null
            # This ensures that the entry will have valid references before being inserted/updated
            if bde_entry["molecule_id"] and bde_entry["fragment_id"]:
                # Upsert (insert or update) the Bde entry into the database
                upsert_record(Bde, bde_entry)
            else:
                # If either molecule_id or fragment_id is missing, log a warning
                logging.warning(f"⚠️ Missing molecule_id or fragment_id for BDE entry: {bde_entry}")
        # --- NOVO BLOCO PARA RADIOSSENSIBILIZADORES ---
    #if "radiossensibilizadores" in sheets:
       # df = sheets["radiossensibilizadores"]
       # for radio in df.dropna(how="all").to_dict(orient="records"):
            # Se no Excel usaste 'id' e 'name', o upsert_record trata de tudo
            # Se usaste o nome para identificar unicamente:
            #upsert_record(radiossensibilizadores, radio, unique_field="id")
    # ----------------------------------------------
   # Commit all changes
    session.commit()
    print("✅ Data successfully inserted/updated in the database!")

except IntegrityError as e:
    session.rollback()
    logging.error(f"❌ Integrity Error: {e}")
    print(f"❌ Integrity error: {e}")

except Exception as e:
    session.rollback()
    logging.error(f"❌ Error inserting/updating data: {e}")
    print(f"❌ Error inserting/updating data: {e}")

finally:
    session.close()
