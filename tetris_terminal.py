# Visual interface in the terminal for tetris_engine.py
import tetris_engine

engine = tetris_engine.tEngine()

while not engine.game_over:
    engine.drop()