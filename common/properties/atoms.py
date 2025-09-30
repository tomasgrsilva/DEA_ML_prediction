class Atom:

    def __init__(self, symbol, name):
        self.symbol = symbol
        self.name = name

    def __repr__(self):
        return self.symbol


Cl = Atom("Cl", "Chlorine")
Br = Atom("Br", "Bromine")
F = Atom("F", "Fluorine")
I = Atom("I", "Iodine")

halogen_symbols = [Cl.symbol, Br.symbol, F.symbol, I.symbol]