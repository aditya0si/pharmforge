"""Tests for data loader."""
from pharmforge.data.loader import load_molecules, search_molecules_sync


def test_load_molecules_count():
    mols = load_molecules()
    assert len(mols) >= 40
    assert len(mols) == 800  # we generate 800

def test_molecules_have_required_fields():
    mols = load_molecules()
    for m in mols[:10]:
        assert m.id.startswith("PF")
        assert m.name
        assert m.smiles
        assert m.description

def test_molecule_ids_unique():
    mols = load_molecules()
    ids = [m.id for m in mols]
    assert len(ids) == len(set(ids))

def test_search_molecules_sync():
    hits = search_molecules_sync("imatinib", limit=5)
    assert len(hits) >= 1
    assert any("imatinib" in h.name.lower() for h in hits)

def test_first_molecule_is_imatinib():
    mols = load_molecules()
    assert mols[0].id == "PF0001"
    assert "imatinib" in mols[0].name.lower()
