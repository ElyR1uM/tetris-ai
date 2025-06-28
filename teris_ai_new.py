"""Convolutional Neural Network for Tetris AI."""
import numpy as np
from tetris_engine import tEngine

class qAgent:
    """Agent to play Tetris"""
    def __init__(self):
        """Initialize the Tetris AI agent."""
        self.engine = tEngine()
        self.state = None
        self.action_space = ['left', 'right', 'drop', 'rotate']