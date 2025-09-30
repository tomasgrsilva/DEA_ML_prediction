from sqlalchemy import Integer, String, Float, ForeignKeyConstraint, JSON, ForeignKey, Table, Column
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql.schema import UniqueConstraint
from deadb.common.database import Base

# Families table divided into metadata tables for each level of the hierarchy

class Level1Category(Base):
    __tablename__ = 'level1_categories'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)  # e.g., "Aliphatic", "Aromatic"
    description: Mapped[str] = mapped_column(String, nullable=True)  # e.g., "Open-chain or non-aromatic cyclic compounds"

    # One-to-many relationship with families
    families: Mapped[list["Family"]] = relationship("Family", back_populates="level1")

class Level2Category(Base):
    __tablename__ = 'level2_categories'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)  # e.g., "Alkanes", "Benzenes"
    description: Mapped[str] = mapped_column(String, nullable=True)  # e.g., "Saturated hydrocarbons with single bonds"

    # Relationships
    families: Mapped[list["Family"]] = relationship("Family", back_populates="level2")

family_level3_link = Table(
    'family_level3_link',
    Base.metadata,
    Column('family_id', Integer, ForeignKey('families.id', ondelete="CASCADE")),
    Column('level3_id', Integer, ForeignKey('level3_categories.id', ondelete="CASCADE")),
    UniqueConstraint('family_id', 'level3_id')
)

class Level3Category(Base):
    __tablename__ = 'level3_categories'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)  # e.g., "Haloalkanes", "Phenols"
    description: Mapped[str] = mapped_column(String, nullable=True)

    families: Mapped[list["Family"]] = relationship(
        "Family",
        secondary=family_level3_link,
        back_populates="level3_categories"
    )


class Level4Category(Base):
    __tablename__ = 'level4_categories'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)  # e.g., "Mono halogenated", "Poly-substituted"
    description: Mapped[str] = mapped_column(String, nullable=True)  # e.g., "One halogen substituent"

    # Relationships
    families: Mapped[list["Family"]] = relationship("Family", back_populates="level4")

class Family(Base):
    __tablename__ = 'families'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    level1_id: Mapped[int] = mapped_column(Integer, ForeignKey('level1_categories.id'), nullable=False)
    level2_id: Mapped[int] = mapped_column(Integer, ForeignKey('level2_categories.id'), nullable=False)

    level4_id: Mapped[int] = mapped_column(Integer, ForeignKey('level4_categories.id'), nullable=True)  # Optional

    # Relationships to metadata tables
    level1: Mapped["Level1Category"] = relationship("Level1Category", back_populates="families")
    level2: Mapped["Level2Category"] = relationship("Level2Category", back_populates="families")

    level3_categories: Mapped[list["Level3Category"]] = relationship(
        "Level3Category",
        secondary=family_level3_link,
        back_populates="families"
    )
    level4: Mapped["Level4Category"] = relationship("Level4Category", back_populates="families")

    # One Family can be linked to multiple Molecules
    molecules: Mapped[list["Molecule"]] = relationship("Molecule", back_populates="family", cascade="all, delete-orphan")


class Molecule(Base):
    __tablename__ = 'molecules'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    formula: Mapped[str] = mapped_column(String, nullable=False)
    family_id: Mapped[int] = mapped_column(Integer, nullable=True)
    molecular_weight: Mapped[float] = mapped_column(Float, nullable=True)

    atomic_composition: Mapped[dict] = mapped_column(JSON, nullable=False)
    halogen_positions: Mapped[dict] = mapped_column(JSON, nullable=True)
    dipole_moment: Mapped[float] = mapped_column(Float, nullable=True)

    smiles: Mapped[str] = mapped_column(String, nullable=True, unique=True)
    casrn: Mapped[str] = mapped_column(String, nullable=True, unique=False)

    # Foreign key with ondelete="CASCADE" handled by ForeignKeyConstraint
    __table_args__ = (
        ForeignKeyConstraint(['family_id'], ['families.id'], ondelete="CASCADE"),
    )

    # Relationships
    papers: Mapped[list["PaperMoleculeLink"]] = relationship("PaperMoleculeLink", back_populates="molecule")

    # Relationship to family (Many Molecules can have the same Family)
    family: Mapped["Family"] = relationship("Family", back_populates="molecules")

    bde_entries: Mapped[list["Bde"]] = relationship("Bde", back_populates="molecule")


class Setup(Base):
    __tablename__ = 'setups'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    exp_setup: Mapped[str] = mapped_column(String, nullable=False)
    setup_explained: Mapped[str] = mapped_column(String, nullable=True)

    # One Setup can be linked to multiple Papers
    papers: Mapped[list["Paper"]] = relationship("Paper", back_populates="setup", cascade="all, delete-orphan")


class Journal(Base):
    __tablename__ = 'journals'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)

    # Optional reverse relationship to Paper
    papers: Mapped[list["Paper"]] = relationship("Paper", back_populates="journal")


# Association table for Paper-Author
paper_author_link = Table(
    'paper_author_link',
    Base.metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('paper_id', Integer, ForeignKey('papers.id', ondelete="CASCADE")),
    Column('author_id', Integer, ForeignKey('authors.id', ondelete="CASCADE")),
    UniqueConstraint('paper_id', 'author_id')
)


# Association table for Paper-Institution
paper_institution_link = Table(
    'paper_institution_link',
    Base.metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('paper_id', Integer, ForeignKey('papers.id', ondelete="CASCADE")),
    Column('institution_id', Integer, ForeignKey('institutions.id', ondelete="CASCADE")),
    UniqueConstraint('paper_id', 'institution_id')
)

class Author(Base):
    __tablename__ = 'authors'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)

    papers: Mapped[list["Paper"]] = relationship(
        "Paper", secondary=paper_author_link, back_populates="authors"
    )

class Institution(Base):
    __tablename__ = 'institutions'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)

    papers: Mapped[list["Paper"]] = relationship(
        "Paper", secondary=paper_institution_link, back_populates="institutions"
    )

class Paper(Base):
    __tablename__ = 'papers'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    publication_year: Mapped[int] = mapped_column(Integer, nullable=False)
    doi: Mapped[str] = mapped_column(String, unique=True, nullable=True)
    url: Mapped[str] = mapped_column(String, nullable=False)
    journal_id: Mapped[int] = mapped_column(Integer, ForeignKey('journals.id'), nullable=True)
    setup_id: Mapped[int] = mapped_column(Integer, ForeignKey('setups.id', ondelete="CASCADE"))

    # Relationships
    authors: Mapped[list["Author"]] = relationship(
        "Author", secondary=paper_author_link, back_populates="papers"
    )
    institutions: Mapped[list["Institution"]] = relationship(
        "Institution", secondary=paper_institution_link, back_populates="papers"
    )
    molecules: Mapped[list["PaperMoleculeLink"]] = relationship("PaperMoleculeLink", back_populates="paper")
    setup: Mapped["Setup"] = relationship("Setup", back_populates="papers")
    journal: Mapped["Journal"] = relationship("Journal", back_populates="papers")

    __table_args__ = (
        ForeignKeyConstraint(['setup_id'], ['setups.id'], ondelete="CASCADE"),
    )


class PaperMoleculeLink(Base):
    __tablename__ = 'paper_molecule_link'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    molecule_id: Mapped[int] = mapped_column(Integer, unique=False)
    paper_id: Mapped[int] = mapped_column(Integer)

    # Foreign keys with ondelete="CASCADE" handled by ForeignKeyConstraint
    __table_args__ = (
        ForeignKeyConstraint(['molecule_id'], ['molecules.id'], ondelete="CASCADE"),
        ForeignKeyConstraint(['paper_id'], ['papers.id'], ondelete="CASCADE"),
    )

    # Relationships (back_populates links the relationship bidirectionally)
    molecule: Mapped["Molecule"] = relationship("Molecule", back_populates="papers")
    paper: Mapped["Paper"] = relationship("Paper", back_populates="molecules")

    # One PaperMoleculeLink can have many Results
    results_list: Mapped[list["Results"]] = relationship("Results", back_populates="paper_molecule_link")


class Fragment(Base):
    __tablename__ = "fragments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    formula: Mapped[str] = mapped_column(String, nullable=False)
    charge: Mapped[int] = mapped_column(Integer, nullable=False)
    molecular_weight: Mapped[float] = mapped_column(Float, nullable=True)
    electron_affinity: Mapped[float] = mapped_column(Float, nullable=True)

    # One Fragment can be referenced by many Results
    results_list: Mapped[list["Results"]] = relationship("Results", back_populates="fragment")

    bde_entries: Mapped[list["Bde"]] = relationship("Bde", back_populates="fragment")


class Results(Base):
    __tablename__ = "results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_molecule_id: Mapped[int] = mapped_column(Integer)
    fragment_id: Mapped[int] = mapped_column(Integer)
    peak: Mapped[float] = mapped_column(Float, nullable=False)
    max_cross_section: Mapped[float] = mapped_column(Float, nullable=True)
    vae: Mapped[float] = mapped_column(Float, nullable=True)
    energy_res: Mapped[float] = mapped_column(Float, nullable=True)
    appearance_energy: Mapped[float] = mapped_column(Float, nullable=True)
    relative_intensity: Mapped[float] = mapped_column(Float, nullable=True)

    # Foreign keys with ondelete="CASCADE" handled by ForeignKeyConstraint
    __table_args__ = (
        ForeignKeyConstraint(['paper_molecule_id'], ['paper_molecule_link.id'], ondelete="CASCADE"),
        ForeignKeyConstraint(['fragment_id'], ['fragments.id'], ondelete="CASCADE"),
    )

    # Relationship to PaperMoleculeLink (many Results can belong to one PaperMoleculeLink)
    paper_molecule_link: Mapped["PaperMoleculeLink"] = relationship("PaperMoleculeLink", back_populates="results_list")

    # Relationship to Fragment (many Results can refer to one Fragment)
    fragment: Mapped["Fragment"] = relationship("Fragment", back_populates="results_list")


class Bde(Base):
    __tablename__ = "bdes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    molecule_id: Mapped[int] = mapped_column(Integer, nullable=False)
    fragment_id: Mapped[int] = mapped_column(Integer, nullable=False)
    bde_values: Mapped[list[float]] = mapped_column(JSON, nullable=False)

    # Foreign keys with cascading deletes
    __table_args__ = (
        ForeignKeyConstraint(['molecule_id'], ['molecules.id'], ondelete="CASCADE"),
        ForeignKeyConstraint(['fragment_id'], ['fragments.id'], ondelete="CASCADE")
    )

    # Relationships
    molecule: Mapped["Molecule"] = relationship("Molecule", back_populates="bde_entries")
    fragment: Mapped["Fragment"] = relationship("Fragment", back_populates="bde_entries")