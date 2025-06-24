#!/usr/bin/env python3
"""
Reward calculation system for Tetris AI training
"""

class RewardCalculator:
    def __init__(self):
        # Reward weights - tune these for different AI behaviors
        self.weights = {
            'survival': 1.0,
            'game_over': -100.0,
            'lines_cleared': {1: 10, 2: 25, 3: 75, 4: 300},
            'efficiency_change': 0.5,
            'score_change': 0.1,
            'height_penalty': -0.2,
            'hole_penalty': -3.0,
            'move_penalty': -0.05  # Small penalty for each move to encourage efficiency
        }
    
    def calculate_reward(self, previous_state, action, new_state, engine):
        """
        Calculate reward for a state transition
        
        Args:
            previous_state: (board_state, piece_info) before action
            action: action taken ('left', 'right', 'down', 'rotate', 'drop')
            new_state: (board_state, piece_info) after action
            engine: tEngine instance for accessing game stats
            
        Returns:
            float: reward value
        """
        reward = 0.0
        
        # 1. Survival reward
        if not engine.game_over:
            reward += self.weights['survival']
        else:
            reward += self.weights['game_over']
            return reward  # End calculation on game over
        
        # 2. Move penalty (encourage efficient play)
        reward += self.weights['move_penalty']
        
        # 3. Line clearing rewards
        lines_cleared = self._count_lines_cleared(previous_state[0], new_state[0])
        if lines_cleared > 0:
            reward += self.weights['lines_cleared'].get(lines_cleared, 0)
        
        # 4. Efficiency-based reward
        if hasattr(engine, 'efficiency_breakdown'):
            prev_efficiency = getattr(engine, 'prev_efficiency', 50.0)  # Default if first move
            current_efficiency = engine.efficiency
            efficiency_change = current_efficiency - prev_efficiency
            reward += efficiency_change * self.weights['efficiency_change']
            
            # Store for next comparison
            engine.prev_efficiency = current_efficiency
            
            # Additional penalties based on efficiency breakdown
            breakdown = engine.efficiency_breakdown
            
            # Height penalty
            if breakdown['max_height'] > 15:  # Penalize very high stacks
                reward += self.weights['height_penalty'] * (breakdown['max_height'] - 15)
            
            # Hole penalty
            if breakdown['holes'] > 0:
                reward += self.weights['hole_penalty'] * breakdown['holes']
        
        # 5. Score improvement (secondary reward)
        score_change = engine.score - getattr(engine, 'prev_score', 0)
        if score_change > 0:
            reward += score_change * self.weights['score_change']
        engine.prev_score = engine.score
        
        return reward
    
    def _count_lines_cleared(self, prev_board, new_board):
        """Count how many lines were cleared between board states"""
        prev_full_lines = sum(1 for row in prev_board if all(cell for cell in row))
        new_full_lines = sum(1 for row in new_board if all(cell for cell in row))
        
        # Lines cleared = difference in filled rows + any that were actually cleared
        # This is approximate - actual line clearing is handled by the engine
        return 0  # Let the engine track this instead
    
    def get_state_features(self, board_state, piece_info):
        """
        Extract numerical features from game state for Q-learning
        Returns a hashable state representation
        
        FIXED: Ensures consistent state representation format
        """
        try:
            board, piece_data = board_state, piece_info
            
            # Validate inputs
            if not isinstance(board, (list, tuple)):
                raise ValueError(f"Invalid board state type: {type(board)}")
            
            if not isinstance(piece_data, (list, tuple)) or len(piece_data) != 4:
                raise ValueError(f"Invalid piece_data format: {piece_data}")
            
            # Calculate board features
            heights = self._get_column_heights(board)
            holes = self._count_holes(board)
            bumpiness = self._calculate_bumpiness(heights)
            
            # Piece features - ensure consistent unpacking
            piece_type, piece_x, piece_y, piece_shape = piece_data
            
            # Validate piece data
            if not isinstance(piece_type, str):
                piece_type = str(piece_type)
            
            if not isinstance(piece_x, int):
                piece_x = int(piece_x)
                
            if not isinstance(piece_y, int):
                piece_y = int(piece_y)
            
            # Create a compact, consistent state representation
            state_features = (
                tuple(heights),     # Column heights (tuple of ints)
                int(holes),         # Number of holes (int)
                int(bumpiness),     # Bumpiness (int, rounded)
                str(piece_type),    # Current piece type (string)
                int(piece_x),       # Piece X position (int)
                int(piece_y)        # Piece Y position (int)
            )
            
            # Validate the final state representation
            if not isinstance(state_features, tuple) or len(state_features) != 6:
                raise ValueError(f"Invalid state_features format: {state_features}")
            
            return state_features
            
        except Exception as e:
            print(f"Error in get_state_features: {e}")
            print(f"Board state: {board_state}")
            print(f"Piece info: {piece_info}")
            # Return a safe default state
            return ((0,) * 10, 0, 0, 'I', 4, 0)
    
    def _get_column_heights(self, board):
        """Get height of each column"""
        try:
            heights = []
            board_height = len(board)
            board_width = len(board[0]) if board else 10
            
            for x in range(board_width):
                height = 0
                for y in range(board_height):
                    if board[y][x] == 1:
                        height = board_height - y
                        break
                heights.append(height)
            return heights
        except Exception as e:
            print(f"Error calculating column heights: {e}")
            return [0] * 10  # Safe default
    
    def _count_holes(self, board):
        """Count holes in the board"""
        try:
            holes = 0
            board_height = len(board)
            board_width = len(board[0]) if board else 10
            
            for x in range(board_width):
                found_block = False
                for y in range(board_height):
                    if board[y][x] == 1:
                        found_block = True
                    elif found_block and board[y][x] == 0:
                        holes += 1
            return holes
        except Exception as e:
            print(f"Error counting holes: {e}")
            return 0  # Safe default
    
    def _calculate_bumpiness(self, heights):
        """Calculate bumpiness (height differences between adjacent columns)"""
        try:
            bumpiness = 0
            for i in range(len(heights) - 1):
                bumpiness += abs(heights[i] - heights[i + 1])
            return bumpiness
        except Exception as e:
            print(f"Error calculating bumpiness: {e}")
            return 0  # Safe default
    
    def update_weights(self, weight_updates):
        """Update reward weights during training"""
        for key, value in weight_updates.items():
            if key in self.weights:
                if isinstance(self.weights[key], dict):
                    self.weights[key].update(value)
                else:
                    self.weights[key] = value