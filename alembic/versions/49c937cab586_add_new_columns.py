"""Add new columns

Revision ID: 49c937cab586
Revises: 465b7ec3f705
Create Date: 2025-03-09 17:52:04.553991

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '49c937cab586'
down_revision: Union[str, None] = '465b7ec3f705'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new columns
    op.add_column('papers', sa.Column('publication_year', sa.Integer(), nullable=True))
    op.add_column('fragments', sa.Column('electron_affinity', sa.Float(), nullable=True))
    op.add_column('families', sa.Column('family_description', sa.String(length=255), nullable=True))
    op.add_column('setups', sa.Column('setup_explained', sa.String(length=255), nullable=True))


def downgrade() -> None:
    # Remove the added columns if we need to rollback
    op.drop_column('papers', 'publication_year')
    op.drop_column('fragments', 'electron_affinity')
    op.drop_column('families', 'family_description')
    op.drop_column('setups', 'setup_explained')
