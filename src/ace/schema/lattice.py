from abc import ABC, abstractmethod


class Lattice(ABC):
    @staticmethod
    @abstractmethod
    def bottom() -> "Lattice":
        pass

    @abstractmethod
    def __add__(self, other: "Lattice") -> "Lattice":
        pass

    @abstractmethod
    def __mul__(self, other: "Lattice") -> "Lattice":
        pass

    @abstractmethod
    def __le__(self, other: "Lattice") -> bool:
        pass

    @abstractmethod
    def __ge__(self, other: "Lattice") -> bool:
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass


class SubsetLattice(Lattice):
    val: set[str]

    @staticmethod
    def bottom() -> "SubsetLattice":
        return SubsetLattice(set())

    def __init__(self, val: set[str]):
        self.val = val

    def __add__(self, other: "Lattice") -> "SubsetLattice":
        if isinstance(other, SubsetLattice):
            return SubsetLattice(self.val | other.val)
        raise TypeError(f"Cannot add SubsetLattice with {type(other)}")

    def __mul__(self, other: "Lattice") -> "SubsetLattice":
        if isinstance(other, SubsetLattice):
            return SubsetLattice(self.val & other.val)
        raise TypeError(f"Cannot multiply SubsetLattice with {type(other)}")

    def __le__(self, other: "Lattice") -> bool:
        if isinstance(other, SubsetLattice):
            return self.val <= other.val
        return False

    def __ge__(self, other: "Lattice") -> bool:
        if isinstance(other, SubsetLattice):
            return self.val >= other.val
        return False

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SubsetLattice):
            return self.val == other.val
        return False

    def __repr__(self) -> str:
        return f"SubsetLattice({self.val})"
