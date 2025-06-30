"""
Optimized Reward Function for Tetris DQN Training
"""

import numpy as np

class RewardCalculator:
    def __init__(self):
        # Reward weights - tune these based on training performance
        self.weights = {
            'lines_cleared': 100,      # Primary objective
            'score_increase': 100,     # Direct score improvement
            'height_penalty': -0.5,    # Penalize tall stacks
            'holes_penalty': -1,      # Strongly penalize holes
            'bumpiness_penalty': -0.1, # Penalize uneven surface
            'well_bonus': 0.5,           # Bonus for creating wells
            'line_clear_bonus': {      # Bonus multipliers for line clears
                1: 40,   # Single
                2: 100,  # Double  
                3: 300,  # Triple
                4: 1200  # Tetris
            },
            'survival_bonus': 1,       # Small bonus for staying alive
            'game_over_penalty': -100, # Heavy penalty for game over
            'move_penalty': 0,      # Small penalty to encourage efficiency
        }
    
    def calculate_reward(self, prev_state, action, new_state, engine):
        """
        Calculate reward based on state transition
        
        Args:
            prev_state: Previous game state (board, piece_info)
            action: Action taken ('left', 'right', 'down', 'rotate', 'drop')
            new_state: New game state after action
            engine: Game engine instance
        """
        reward = 0
        
        # Extract states
        prev_board, prev_piece_info = prev_state
        new_board, new_piece_info = new_state
        
        # 1. Game Over Penalty
        if engine.game_over:
            reward += self.weights['game_over_penalty']
            return reward  # Return immediately on game over
        
        # 2. Lines Cleared Reward (most important)
        lines_cleared = self._count_lines_cleared(prev_board, new_board)
        if lines_cleared > 0:
            # Base reward for lines
            reward += lines_cleared * self.weights['lines_cleared']
            # Bonus for multiple lines (encourage Tetris)
            if lines_cleared in self.weights['line_clear_bonus']:
                reward += self.weights['line_clear_bonus'][lines_cleared]
        
        # 3. Score Increase
        if hasattr(engine, 'score') and hasattr(engine, 'prev_score'):
            score_diff = engine.score - getattr(engine, 'prev_score', 0)
            reward += score_diff * self.weights['score_increase']
        
        # 4. Board Analysis Penalties/Bonuses
        board_metrics = self._analyze_board(new_board)
        
        # Height penalty (encourage keeping board low)
        reward += board_metrics['max_height'] * self.weights['height_penalty']
        
        # Holes penalty (strongly discourage creating holes)
        reward += board_metrics['holes'] * self.weights['holes_penalty']
        
        # Bumpiness penalty (encourage flat surface)
        reward += board_metrics['bumpiness'] * self.weights['bumpiness_penalty']
        
        # Well bonus (encourage creating wells for line clears)
        reward += board_metrics['wells'] * self.weights['well_bonus']
        
        # 5. Survival bonus (small reward for staying alive)
        reward += self.weights['survival_bonus']
        
        # 6. Move efficiency penalty
        reward += self.weights['move_penalty']
        
        # 7. Special action bonuses
        reward += self._action_specific_rewards(action, prev_board, new_board)
        
        return reward
    
    def _count_lines_cleared(self, prev_board, new_board):
        """Count how many lines were cleared"""
        prev_full_lines = sum(1 for row in prev_board if all(cell != 0 and cell != ' ' for cell in row))
        new_full_lines = sum(1 for row in new_board if all(cell != 0 and cell != ' ' for cell in row))
        
        # Lines cleared = reduction in full lines + any new empty rows at top
        prev_empty_top = sum(1 for row in prev_board if all(cell == 0 or cell == ' ' for cell in row))
        new_empty_top = sum(1 for row in new_board if all(cell == 0 or cell == ' ' for cell in row))
        
        # Estimate lines cleared (this is approximate due to board representation)
        return max(0, new_empty_top - prev_empty_top)
    
    def _analyze_board(self, board):
        """Analyze board for various metrics"""
        metrics = {
            'max_height': 0,
            'holes': 0,
            'bumpiness': 0,
            'wells': 0
        }
        
        heights = []
        
        # Calculate column heights and find holes
        for col in range(len(board[0])):
            height = 0
            holes_in_col = 0
            found_block = False
            
            # Scan from top to bottom
            for row in range(len(board)):
                cell = board[row][col]
                if cell != 0 and cell != ' ':
                    if not found_block:
                        height = len(board) - row
                        found_block = True
                else:
                    # Empty cell - check if it's a hole
                    if found_block:
                        holes_in_col += 1
            
            heights.append(height)
            metrics['holes'] += holes_in_col
        
        metrics['max_height'] = max(heights) if heights else 0
        
        # Calculate bumpiness (difference between adjacent columns)
        for i in range(len(heights) - 1):
            metrics['bumpiness'] += abs(heights[i] - heights[i + 1])
        
        # Calculate wells (columns significantly lower than neighbors)
        for i in range(len(heights)):
            left_height = heights[i - 1] if i > 0 else 0
            right_height = heights[i + 1] if i < len(heights) - 1 else 0
            current_height = heights[i]
            
            # Well depth
            well_depth = min(left_height, right_height) - current_height
            if well_depth > 0:
                metrics['wells'] += well_depth
        
        return metrics
    
    def _action_specific_rewards(self, action, prev_board, new_board):
        """Rewards/penalties for specific actions"""
        reward = 0
        
        if action == 'drop':
            # Small bonus for hard drops (encourages decisive play)
            reward += 2
        elif action == 'rotate':
            # Small penalty for rotations (discourage excessive rotation)
            reward -= 0.5
        elif action == 'down':
            # Very small bonus for soft drops
            reward += 0.1
        
        return reward
    
    def adjust_weights(self, performance_metrics):
        """
        Dynamically adjust weights based on training performance
        Call this periodically during training to optimize rewards
        """
        avg_score = performance_metrics.get('avg_score', 0)
        avg_lines = performance_metrics.get('avg_lines', 0)
        game_over_rate = performance_metrics.get('game_over_rate', 1.0)
        
        # If agent is dying too quickly, reduce penalties
        if game_over_rate > 0.9:
            self.weights['height_penalty'] *= 0.9
            self.weights['holes_penalty'] *= 0.9
            self.weights['survival_bonus'] *= 1.1
        
        # If agent is clearing lines well, increase line clear bonuses
        if avg_lines > 1:
            for key in self.weights['line_clear_bonus']:
                self.weights['line_clear_bonus'][key] *= 1.05
        
        # If scores are improving, balance exploration vs exploitation
        if avg_score > 1000:
            self.weights['move_penalty'] *= 0.95  # Reduce move penalty
    
    def get_reward_breakdown(self, prev_state, action, new_state, engine):
        """
        Get detailed breakdown of reward components for debugging
        """
        breakdown = {}
        
        prev_board, prev_piece_info = prev_state
        new_board, new_piece_info = new_state
        
        if engine.game_over:
            breakdown['game_over'] = self.weights['game_over_penalty']
            return breakdown
        
        # Calculate each component
        lines_cleared = self._count_lines_cleared(prev_board, new_board)
        if lines_cleared > 0:
            breakdown['lines_cleared'] = lines_cleared * self.weights['lines_cleared']
            if lines_cleared in self.weights['line_clear_bonus']:
                breakdown['line_clear_bonus'] = self.weights['line_clear_bonus'][lines_cleared]
        
        board_metrics = self._analyze_board(new_board)
        breakdown['height_penalty'] = board_metrics['max_height'] * self.weights['height_penalty']
        breakdown['holes_penalty'] = board_metrics['holes'] * self.weights['holes_penalty']
        breakdown['bumpiness_penalty'] = board_metrics['bumpiness'] * self.weights['bumpiness_penalty']
        breakdown['well_bonus'] = board_metrics['wells'] * self.weights['well_bonus']
        breakdown['survival_bonus'] = self.weights['survival_bonus']
        breakdown['move_penalty'] = self.weights['move_penalty']
        breakdown['action_specific'] = self._action_specific_rewards(action, prev_board, new_board)
        
        return breakdown