from game import Game
from Strategies.m1strategies import *

titForTat = TitForTat()
alwaysDefect = AlwaysDefect()
wsls = WinStayLoseShift()
ran = RandomStrategy()

game1 = Game(wsls,ran,100,0.1,'CC')
game1.run()
game1.printResults()
