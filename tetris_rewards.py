"""
Enhanced reward calculation system for Tetris AI training
Focuses on line clearing behavior and strategic positioning
"""
from tetris_engine import tEngine

class RewardCalculator:
    def __init__(self):
        self.engine = tEngine()
        
        # Dramatically improved reward weights
        self.weights = {
            'survival': 0.1,  # Reduced - survival alone isn't enough
            'game_over': -500.0,  # Reduced penalty to encourage exploration
            
            # MASSIVELY increased line clearing rewards
            'lines_cleared': {
                1: 100,    # Single line - good reward
                2: 300,    # Double - much better
                3: 800,    # Triple - excellent
                4: 2000    # Tetris - outstanding!
            },
            
            # Strategic positioning rewards
            'line_completion_potential': 5.0,   # Reward for almost completing lines
            'well_formation': 2.0,              # Reward for creating wells for I-pieces
            'piece_placement_efficiency': 1.0,   # Reward for good piece placement
            
            # Penalties (reduced to not dominate rewards)
            'height_penalty': -0.1,             # Much smaller penalty
            'hole_penalty': -1.0,               # Reduced hole penalty
            'bumpiness_penalty': -0.05,         # Small bumpiness penalty
            'move_penalty': -0.01,              # Tiny move penalty
            
            # New strategic rewards
            'height_reduction': 3.0,            # Reward for reducing max height
            'even_height': 0.5,                 # Reward for keeping heights even
        }
        
        # Track previous state for comparisons
        self.prev_max_height = 0
        self.prev_total_holes = 0
        self.prev_lines_cleared = 0
    
    def calculate_reward(self, previous_state, action, new_state, engine):
        """
        Enhanced reward calculation focusing on line clearing
        """
        reward = 0.0
        
        # 1. Game over penalty (but not too harsh to allow exploration)
        if engine.game_over:
            reward += self.weights['game_over']
            return reward
        
        # 2. Survival reward (small)
        reward += self.weights['survival']
        
        # 3. MAJOR LINE CLEARING REWARDS
        lines_cleared_this_move = engine.total_cleared - self.prev_lines_cleared
        if lines_cleared_this_move > 0:
            line_reward = self.weights['lines_cleared'].get(lines_cleared_this_move, 0)
            reward += line_reward
            print(f"LINE CLEAR! {lines_cleared_this_move} lines = +{line_reward} reward")
        
        # 4. Line completion potential - reward for almost completing lines
        reward += self._calculate_line_potential_reward(new_state[0])
        
        # 5. Well formation reward - encourage creating wells for I-pieces
        reward += self._calculate_well_reward(new_state[0])
        
        # 6. Height management rewards
        current_heights = self._get_column_heights(new_state[0])
        max_height = max(current_heights)
        
        # Reward for reducing maximum height
        if max_height < self.prev_max_height:
            reward += self.weights['height_reduction'] * (self.prev_max_height - max_height)
        
        # Small penalty for excessive height
        if max_height > 15:
            reward += self.weights['height_penalty'] * (max_height - 15)
        
        # 7. Hole penalties (but not too harsh)
        holes = self._count_holes(new_state[0])
        new_holes = holes - self.prev_total_holes
        if new_holes > 0:
            reward += self.weights['hole_penalty'] * new_holes
        
        # 8. Bumpiness penalty (encourage smoother tops)
        bumpiness = self._calculate_bumpiness(current_heights)
        reward += self.weights['bumpiness_penalty'] * bumpiness
        
        # 9. Small move penalty to encourage efficiency
        reward += self.weights['move_penalty']
        
        # Update tracking variables
        self.prev_max_height = max_height
        self.prev_total_holes = holes
        self.prev_lines_cleared = engine.total_cleared
        
        return reward
    
    def _calculate_line_potential_reward(self, board):
        """
        Reward for lines that are almost complete
        This encourages the AI to work towards line clearing
        """
        reward = 0.0
        board_height = len(board)
        board_width = len(board[0]) if board else 10
        
        for y in range(board_height):
            filled_cells = sum(1 for x in range(board_width) if board[y][x] == 1)
            
            # Reward based on how close to completion each line is
            if filled_cells >= 7:  # 7+ out of 10 cells filled
                completion_ratio = filled_cells / board_width
                reward += self.weights['line_completion_potential'] * completion_ratio
        
        return reward
    
    def _calculate_well_reward(self, board):
        """
        Reward for creating wells that can accommodate I-pieces (4-block verticals)
        """
        reward = 0.0
        heights = self._get_column_heights(board)
        
        # Look for wells (columns significantly lower than neighbors)
        for i in range(len(heights)):
            left_height = heights[i-1] if i > 0 else heights[i]
            right_height = heights[i+1] if i < len(heights)-1 else heights[i]
            current_height = heights[i]
            
            # If this column is 3+ blocks lower than both neighbors, it's a good well
            well_depth = min(left_height - current_height, right_height - current_height)
            if well_depth >= 3:
                reward += self.weights['well_formation'] * (well_depth / 4.0)  # Normalize
        
        return reward
    
    def get_state_features(self, board_state, piece_info):
        """
        Enhanced state representation focusing on line-clearing opportunities
        """
        try:
            board, piece_data = board_state, piece_info
            
            # Basic features
            heights = self._get_column_heights(board)
            holes = self._count_holes(board)
            bumpiness = self._calculate_bumpiness(heights)
            
            # Enhanced features for line clearing
            line_completion_scores = self._get_line_completion_scores(board)
            well_positions = self._get_well_positions(heights)
            max_height = max(heights)
            height_variance = self._calculate_height_variance(heights)
            
            # Piece features
            piece_type, piece_x, piece_y, piece_shape = piece_data
            
            # Create comprehensive state representation
            state_features = (
                tuple(heights[:10]),  # Column heights (ensure max 10)
                int(holes),
                int(bumpiness),
                int(max_height),
                int(height_variance),
                tuple(line_completion_scores[:5]),  # Top 5 line completion scores
                tuple(well_positions[:3]),  # Top 3 well positions
                str(piece_type),
                int(piece_x),
                min(int(piece_y), 20)  # Cap piece_y to prevent explosion
            )
            
            return state_features
            
        except Exception as e:
            print(f"Error in enhanced state features: {e}")
            # Safe fallback
            return ((0,) * 10, 0, 0, 0, 0, (0,) * 5, (0,) * 3, 'I', 4, 0)
    
    def _get_line_completion_scores(self, board):
        """Get completion scores for each line (how close to being cleared)"""
        board_height = len(board)
        board_width = len(board[0]) if board else 10
        scores = []
        
        for y in range(board_height):
            filled = sum(1 for x in range(board_width) if board[y][x] == 1)
            scores.append(filled)
        
        # Return top 5 scores (most filled lines)
        scores.sort(reverse=True)
        return scores[:5] + [0] * (5 - len(scores[:5]))
    
    def _get_well_positions(self, heights):
        """Get positions and depths of wells"""
        wells = []
        
        for i in range(len(heights)):
            left_height = heights[i-1] if i > 0 else heights[i]
            right_height = heights[i+1] if i < len(heights)-1 else heights[i]
            current_height = heights[i]
            
            well_depth = min(left_height - current_height, right_height - current_height)
            if well_depth > 0:
                wells.append(well_depth)
            else:
                wells.append(0)
        
        # Return top 3 well depths
        wells.sort(reverse=True)
        return wells[:3] + [0] * (3 - len(wells[:3]))
    
    def _calculate_height_variance(self, heights):
        """Calculate variance in column heights"""
        if not heights:
            return 0
        avg_height = sum(heights) / len(heights)
        variance = sum((h - avg_height) ** 2 for h in heights) / len(heights)
        return int(variance)
    
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
        except Exception:
            return [0] * 10
    
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
        except Exception:
            return 0
    
    def _calculate_bumpiness(self, heights):
        """Calculate bumpiness (height differences)"""
        try:
            bumpiness = 0
            for i in range(len(heights) - 1):
                bumpiness += abs(heights[i] - heights[i + 1])
            return bumpiness
        except Exception:
            return 0