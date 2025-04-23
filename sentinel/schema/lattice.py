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
    def __eq__(self, other: "Lattice") -> bool:
        pass


class SubsetLattice():
    val: set[str]

    def bottom() -> "SubsetLattice":
        return SubsetLattice(set())

    def __init__(self, val: set[str]):
        self.val = val

    def __add__(self, other: "SubsetLattice") -> "SubsetLattice":
        return SubsetLattice(self.val | other.val)

    def __mul__(self, other: "SubsetLattice") -> "SubsetLattice":
        return SubsetLattice(self.val & other.val)

    def __le__(self, other: "SubsetLattice") -> bool:
        return self.val <= other.val

    def __ge__(self, other: "SubsetLattice") -> bool:
        return self.val >= other.val

    def __eq__(self, other: "SubsetLattice") -> bool:
        return self.val == other.val

    def __repr__(self) -> str:
        return f"SubsetLattice({self.val})"
