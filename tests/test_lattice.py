from src.ace.schema.lattice import SubsetLattice


def test_subset_lattice_ops():
    a = SubsetLattice({"a", "b"})
    b = SubsetLattice({"b", "c"})
    assert (a + b) == SubsetLattice({"a", "b", "c"})
    assert (a * b) == SubsetLattice({"b"})
    assert a <= SubsetLattice({"a", "b", "c"})
    assert not a <= SubsetLattice({"b"})
    assert a == SubsetLattice({"a", "b"})
    assert repr(a) == "SubsetLattice({'a', 'b'})" or repr(a) == "SubsetLattice({'b', 'a'})"
