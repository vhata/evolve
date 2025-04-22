import pygame
import random
import math
import numpy as np
import time
from typing import List, Tuple, Dict

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
POPULATION_SIZE = 50
FPS = 120  # Frame rate for simulation
FAST_MODE = True  # Toggle to run simulation faster with minimal visuals
MUTATION_RATE = 0.1
LARGE_MUTATION_PROBABILITY = 0.05
FOOD_COUNT = 20
OBSTACLE_COUNT = 10
MAX_POPULATION = 100  # Maximum number of organisms allowed
REPRODUCTION_ENERGY = 800  # Energy threshold for reproduction
FOOD_RESPAWN_RATE = 0.02  # Probability of new food appearing each frame
OBSTACLE_MOVE_CHANCE = 0.005  # Probability of obstacle moving each frame
OBSTACLE_SPEED = 1.0  # Speed at which obstacles move
CORPSE_DECAY_TIME = 500  # How long dead organisms stay visible
CORPSE_TO_FOOD_TIME = 350  # How long until a corpse becomes food
OBSTACLE_EROSION_AMOUNT = 2  # How much obstacles shrink when hit
POOP_CHANCE = 0.1  # Chance to create a small obstacle after eating
POOP_SIZE_RANGE = (5, 10)  # Size range for poop obstacles
SPECIES_SIMILARITY_THRESHOLD = 0.15  # Maximum genome difference for same species
MAX_TRACKED_SPECIES = 6  # Maximum number of species to track and display

# Food type constants
class FoodType:
    STANDARD = 0
    RICH = 1
    SUPERFOOD = 2
    JUNK = 3

# Food type probabilities (must sum to 1.0)
FOOD_TYPE_PROBS = {
    FoodType.STANDARD: 0.6,    # Common, average nutrition
    FoodType.RICH: 0.25,       # Somewhat rare, high nutrition
    FoodType.SUPERFOOD: 0.05,  # Very rare, extremely nutritious
    FoodType.JUNK: 0.1         # Common but low nutrition
}

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 200, 0)  # Darker green for standard food
BLUE = (50, 100, 255)  # Lighter blue for organisms
YELLOW = (255, 255, 0)
GRAY = (80, 80, 80)  # Darker gray for obstacles
BROWN = (139, 69, 19)  # Brown for obstacles

# Food colors by type
FOOD_COLORS = {
    FoodType.STANDARD: (0, 200, 0),       # Green
    FoodType.RICH: (255, 215, 0),         # Gold
    FoodType.SUPERFOOD: (148, 0, 211),    # Purple
    FoodType.JUNK: (169, 169, 169)        # Light gray
}

# Food type names for display
FOOD_TYPE_NAMES = {
    FoodType.STANDARD: "Standard",
    FoodType.RICH: "Rich",
    FoodType.SUPERFOOD: "Super",
    FoodType.JUNK: "Junk"
}

# Species colors - used to clearly distinguish different species
SPECIES_COLORS = [
    (255, 0, 0),    # Red
    (0, 0, 255),    # Blue
    (0, 200, 0),    # Green
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (0, 255, 255),  # Cyan
    (255, 255, 0),  # Yellow
    (255, 105, 180) # Pink
]

class Food:
    def __init__(self, position=None, food_type=None):
        if position:
            self.position = position
        else:
            self.position = (
                random.randint(50, WINDOW_WIDTH - 50),
                random.randint(50, WINDOW_HEIGHT - 50)
            )
        
        # Assign random food type based on probabilities if not specified
        if food_type is None:
            # Use weighted random selection based on probabilities
            r = random.random()
            cumulative = 0
            for food_type, prob in FOOD_TYPE_PROBS.items():
                cumulative += prob
                if r <= cumulative:
                    self.food_type = food_type
                    break
            else:
                # Fallback in case probabilities don't sum to 1.0
                self.food_type = FoodType.STANDARD
        else:
            self.food_type = food_type
            
        # Set base properties according to food type
        self.radius = 5
        self.active = True
        self.age = 0
        
        # Set energy value based on food type
        if self.food_type == FoodType.STANDARD:
            self.energy = 100
            self.max_energy = 200
            self.growth_rate = 10  # Energy added every 50 frames
        elif self.food_type == FoodType.RICH:
            self.energy = 200
            self.max_energy = 300
            self.growth_rate = 15
        elif self.food_type == FoodType.SUPERFOOD:
            self.energy = 400
            self.max_energy = 500
            self.growth_rate = 20
        elif self.food_type == FoodType.JUNK:
            self.energy = 50
            self.max_energy = 70
            self.growth_rate = 5
        
    def update(self):
        # Food becomes more nutritious as it ages (simulating growth)
        if self.active and self.age < 300:  # Cap at 300 frames
            self.age += 1
            if self.age % 50 == 0 and self.energy < self.max_energy:
                self.energy += self.growth_rate
            
    def draw(self, surface):
        if self.active:
            # Get base color from food type
            base_color = FOOD_COLORS[self.food_type]
            
            # Adjust size based on age/ripeness
            size_factor = min(1.5, 1.0 + self.age/300)
            radius = int(self.radius * size_factor)
            
            # Calculate color shade based on age (gets more intense as it ripens)
            ripeness = min(1.0, self.age / 200)
            color = (
                max(0, min(255, int(base_color[0] * (0.7 + 0.3 * ripeness)))),
                max(0, min(255, int(base_color[1] * (0.7 + 0.3 * ripeness)))),
                max(0, min(255, int(base_color[2] * (0.7 + 0.3 * ripeness))))
            )
            
            pygame.draw.circle(surface, color, self.position, radius)
            
            # Add visual indicators for special food types
            if self.food_type == FoodType.RICH or self.food_type == FoodType.SUPERFOOD:
                # Add a glow effect (smaller circles around the main one)
                glow_radius = radius + 2
                glow_color = (color[0], color[1], color[2], 100)  # Semi-transparent
                glow_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, glow_color, self.position, glow_radius)
                surface.blit(glow_surf, (0, 0))
                
                # Add some sparkles for superfood
                if self.food_type == FoodType.SUPERFOOD:
                    for i in range(3):
                        angle = random.uniform(0, 2 * math.pi)
                        dist = radius * 0.8
                        sparkle_x = self.position[0] + math.cos(angle) * dist
                        sparkle_y = self.position[1] + math.sin(angle) * dist
                        pygame.draw.circle(surface, WHITE, (int(sparkle_x), int(sparkle_y)), 1)
            
            # Add standard stem for all food types
            pygame.draw.line(surface, BROWN, 
                          (self.position[0], self.position[1] - radius),
                          (self.position[0] + 3, self.position[1] - radius - 3), 2)

class Obstacle:
    def __init__(self, position=None, poop=False):
        # Position can be specified (for poop obstacles)
        if position:
            self.position = position
        else:
            self.position = (
                random.randint(50, WINDOW_WIDTH - 50),
                random.randint(50, WINDOW_HEIGHT - 50)
            )
            
        # Size depends on type (poop or regular)
        if poop:
            self.radius = random.randint(POOP_SIZE_RANGE[0], POOP_SIZE_RANGE[1])
            self.stationary = True  # Poop doesn't move initially
            self.is_poop = True  # Mark as poop for visual difference
        else:
            self.radius = random.randint(20, 40)
            self.stationary = random.random() < 0.6  # 60% chance to be stationary
            self.is_poop = False
            
        self.direction = random.uniform(0, 2 * math.pi)
        self.speed = 0 
        self.initial_radius = self.radius  # Track original size for visual effects
        self.eroded = False  # Track if obstacle has been eroded
        self.min_radius = 5 if poop else 8  # Minimum size before disappearing
        self.age = 0  # Track age for poop objects (for color change)
        
    def erode(self):
        # Reduce size when hit
        self.radius = max(self.min_radius, self.radius - OBSTACLE_EROSION_AMOUNT)
        self.eroded = True
        
        # Increase speed slightly if mobile obstacle is hit (broken pieces move faster)
        if not self.stationary:
            self.speed = min(OBSTACLE_SPEED * 1.5, self.speed * 1.2)
            
        # Return True if obstacle should be removed
        return self.radius <= self.min_radius
        
    def update(self):
        # Age the obstacle (affects poop appearance)
        self.age += 1
        
        # Special behavior for poop obstacles
        if self.is_poop:
            # Poop has a chance to start moving after it's been around for a while
            if self.stationary and self.age > 100 and random.random() < OBSTACLE_MOVE_CHANCE * 2:
                self.stationary = False
                self.speed = OBSTACLE_SPEED * 0.5  # Slower than regular obstacles
                
            # Poop gradually hardens/dries out, becoming more like a regular obstacle
            if self.age > 300 and random.random() < 0.01:
                self.is_poop = False
                
        # Occasionally change direction or start/stop moving
        if random.random() < OBSTACLE_MOVE_CHANCE:
            if self.stationary:
                # Small chance for a stationary obstacle to start moving
                if random.random() < 0.2:
                    self.stationary = False
                    self.speed = OBSTACLE_SPEED * random.uniform(0.5, 1.0)
                    self.direction = random.uniform(0, 2 * math.pi)
            else:
                # Moving obstacle changes direction or stops
                if random.random() < 0.3:  # 30% chance to stop
                    self.stationary = True
                    self.speed = 0
                else:
                    # Change direction slightly
                    self.direction += random.uniform(-0.5, 0.5)
                    
        # Occasionally eroded obstacles crumble more
        if self.eroded and random.random() < 0.001:
            self.radius = max(self.min_radius, self.radius - 1)
                
        # Move if not stationary
        if not self.stationary:
            new_x = self.position[0] + math.cos(self.direction) * self.speed
            new_y = self.position[1] + math.sin(self.direction) * self.speed
            
            # Boundary checks and bouncing
            if new_x < self.radius or new_x > WINDOW_WIDTH - self.radius:
                self.direction = math.pi - self.direction
                new_x = max(self.radius, min(WINDOW_WIDTH - self.radius, new_x))
                
            if new_y < self.radius or new_y > WINDOW_HEIGHT - self.radius:
                self.direction = -self.direction
                new_y = max(self.radius, min(WINDOW_HEIGHT - self.radius, new_y))
                
            self.position = (new_x, new_y)
        
    def draw(self, surface):
        # Skip if too small
        if self.radius <= self.min_radius:
            return
            
        # Determine color based on type and state
        if self.is_poop:
            # Poop starts brown and gradually darkens/hardens
            age_factor = min(1.0, self.age / 300)
            
            # Fresh poop is more brown, older is darker
            rock_color = (
                max(60, min(139, int(139 - 50 * age_factor))),  # R
                max(30, min(69, int(69 - 20 * age_factor))),     # G 
                max(10, min(19, int(19 - 5 * age_factor)))       # B
            )
        else:
            # Regular obstacle
            rock_color = BROWN
            
            # If eroded, make color slightly different
            if self.eroded:
                # More grey as it erodes
                erosion_factor = (self.initial_radius - self.radius) / (self.initial_radius - self.min_radius)
                rock_color = (
                    min(255, BROWN[0] + int(50 * erosion_factor)),
                    min(255, BROWN[1] + int(50 * erosion_factor)),
                    min(255, BROWN[2] + int(50 * erosion_factor))
                )
        
        # Draw the main obstacle shape
        pygame.draw.circle(surface, rock_color, (int(self.position[0]), int(self.position[1])), self.radius)
        
        # Different textures based on type
        if self.is_poop:
            # Poop has a more irregular texture (small bumps)
            for i in range(min(6, int(self.radius/2))):
                angle = i * math.pi * 2 / min(6, int(self.radius/2))
                bump_dist = self.radius * 0.7
                bump_x = self.position[0] + math.cos(angle) * bump_dist
                bump_y = self.position[1] + math.sin(angle) * bump_dist
                bump_size = max(1, int(self.radius * 0.25))
                
                # Make bumps slightly darker
                bump_color = (
                    max(20, rock_color[0] - 20),
                    max(10, rock_color[1] - 10),
                    max(5, rock_color[2] - 5)
                )
                pygame.draw.circle(surface, bump_color, (int(bump_x), int(bump_y)), bump_size)
        else:
            # Regular obstacle has rock-like shading
            pygame.draw.arc(surface, GRAY, 
                          (int(self.position[0] - self.radius), int(self.position[1] - self.radius),
                           self.radius * 2, self.radius * 2),
                          math.pi/4, math.pi, max(2, self.radius//2))
                      
        # Add cracks if eroded
        if self.eroded:
            # Draw some random cracks
            crack_start = self.position
            erosion_factor = (self.initial_radius - self.radius) / (self.initial_radius - self.min_radius)
            for _ in range(min(5, int(erosion_factor * 10))):
                crack_length = random.uniform(0.3, 0.9) * self.radius
                crack_angle = random.uniform(0, 2 * math.pi)
                crack_end = (
                    crack_start[0] + math.cos(crack_angle) * crack_length,
                    crack_start[1] + math.sin(crack_angle) * crack_length
                )
                pygame.draw.line(surface, (50, 50, 50), 
                              (int(crack_start[0]), int(crack_start[1])),
                              (int(crack_end[0]), int(crack_end[1])), 1)
                      
        # Add a small indicator if the obstacle is mobile
        if not self.stationary:
            indicator_x = self.position[0] + math.cos(self.direction) * (self.radius * 0.7)
            indicator_y = self.position[1] + math.sin(self.direction) * (self.radius * 0.7)
            indicator_color = (200, 100, 50) if not self.is_poop else (160, 80, 40)
            pygame.draw.circle(surface, indicator_color, (int(indicator_x), int(indicator_y)), 
                             max(2, min(3, int(self.radius / 4))))
        
    def collides_with(self, x, y, radius):
        dx = self.position[0] - x
        dy = self.position[1] - y
        distance = math.sqrt(dx*dx + dy*dy)
        return distance < (self.radius + radius)

class Organism:
    # Class variable to assign and track species IDs
    next_species_id = 1
    species_registry = {}  # Map species_id to stats like count, avg fitness, etc.
    
    def __init__(self, genome=None, position=None, parent=None):
        # Position and physics
        if position:
            self.position = position
        else:
            self.position = (
                random.randint(50, WINDOW_WIDTH - 50),
                random.randint(50, WINDOW_HEIGHT - 50)
            )
        self.radius = 10
        self.direction = random.uniform(0, 2 * math.pi)
        self.speed = 1.0  # Start with some speed instead of 0
        self.max_speed = 5
        self.energy = 500
        self.food_eaten = 0
        self.alive = True
        self.age = 0  # Age in simulation steps
        self.reproduction_cooldown = 0  # Cooldown before reproducing again
        self.time_of_death = None  # When the organism died (for decaying corpses)
        self.decomposing = False  # Whether the corpse is turning into food
        self.decompose_timer = 0  # Track decomposition progress
        self.generation = 1 if parent is None else parent.generation + 1  # Track organism generation
        
        # Food preferences - track what this organism has eaten
        self.food_type_eaten = {
            FoodType.STANDARD: 0,
            FoodType.RICH: 0,
            FoodType.SUPERFOOD: 0,
            FoodType.JUNK: 0
        }
        
        # Species identification
        self.parent_species_id = parent.species_id if parent else None
        self.species_id = None  # To be assigned after genome creation
        
        # Individual color variation
        self.color_variation = (
            random.randint(-20, 20),
            random.randint(-20, 20),
            random.randint(-20, 20)
        )
        
        # Genome defines the organism's behavior
        if genome is None:
            self.genome = {
                # Sensors
                'vision_range': random.uniform(80, 200),
                'field_of_view': random.uniform(math.pi/3, math.pi),
                'num_sensors': random.randint(3, 7),
                
                # Behavior
                'turn_factor': random.uniform(0.1, 0.5),
                'speed_factor': random.uniform(0.8, 2.0),
                'metabolism': random.uniform(0.1, 0.3),
                
                # Neural weights (simplified)
                'food_attraction': random.uniform(0.5, 2.0),
                'obstacle_avoidance': random.uniform(0.5, 2.0),
                'exploration_drive': random.uniform(0.3, 1.0),
                
                # Food preferences (attraction to each food type)
                'standard_food_pref': random.uniform(0.5, 1.5),
                'rich_food_pref': random.uniform(0.5, 1.5),
                'superfood_pref': random.uniform(0.5, 1.5),
                'junk_food_pref': random.uniform(0.5, 1.5),
                
                # New traits for continuous evolution
                'reproduction_threshold': random.uniform(700, 900),  # Energy needed to reproduce
                'offspring_energy': random.uniform(0.2, 0.4),  # Percentage of energy to give to offspring
                'size_factor': random.uniform(0.8, 1.2)  # Size of organism (affects feeding and collision)
            }
        else:
            self.genome = genome
            
        # Apply size factor to radius
        self.radius = self.radius * self.genome['size_factor']
        
        # Calculate base fitness
        self.fitness = 0
        
        # Assign a species ID
        self.determine_species()
            
    def sense_environment(self, foods, obstacles):
        # Return a list of sensory inputs based on vision
        vision_range = self.genome['vision_range']
        field_of_view = self.genome['field_of_view']
        num_sensors = int(self.genome['num_sensors'])
        
        # Initialize sensors with food signals by type and obstacle signal
        # Format: (standard_food, rich_food, superfood, junk_food, obstacle)
        sensors = [(0, 0, 0, 0, 0) for _ in range(num_sensors)]
        
        # Calculate sensor angles
        sensor_angles = []
        for i in range(num_sensors):
            angle = self.direction - field_of_view/2 + i * field_of_view/(num_sensors-1 if num_sensors > 1 else 1)
            sensor_angles.append(angle)
            
        # Detect food
        for i, angle in enumerate(sensor_angles):
            # Cast rays and check for intersections
            ray_x = self.position[0] + math.cos(angle) * vision_range
            ray_y = self.position[1] + math.sin(angle) * vision_range
            
            # Initialize closest food distances for each type
            closest_food_dist = {
                FoodType.STANDARD: vision_range,
                FoodType.RICH: vision_range,
                FoodType.SUPERFOOD: vision_range,
                FoodType.JUNK: vision_range
            }
            
            # Check for food along the ray
            for food in foods:
                if food.active:
                    food_dist = math.sqrt((food.position[0] - self.position[0])**2 + 
                                        (food.position[1] - self.position[1])**2)
                    
                    if food_dist < closest_food_dist[food.food_type]:
                        # Check if food is in the direction of the ray
                        food_angle = math.atan2(food.position[1] - self.position[1], 
                                              food.position[0] - self.position[0])
                        angle_diff = abs(angle - food_angle)
                        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
                        
                        if angle_diff < field_of_view / num_sensors:
                            closest_food_dist[food.food_type] = food_dist
                            
                            # Update sensor value for this food type
                            food_signal = 1 - food_dist/vision_range
                            
                            # Create updated sensor tuple with new value for detected food type
                            sensor_values = list(sensors[i])
                            if food.food_type == FoodType.STANDARD:
                                sensor_values[0] = food_signal
                            elif food.food_type == FoodType.RICH:
                                sensor_values[1] = food_signal
                            elif food.food_type == FoodType.SUPERFOOD:
                                sensor_values[2] = food_signal
                            elif food.food_type == FoodType.JUNK:
                                sensor_values[3] = food_signal
                            
                            # Keep obstacle value the same
                            sensors[i] = tuple(sensor_values)
            
            # Check for obstacles along the ray
            closest_obstacle_dist = vision_range
            for obstacle in obstacles:
                obstacle_dist = math.sqrt((obstacle.position[0] - self.position[0])**2 + 
                                        (obstacle.position[1] - self.position[1])**2)
                
                if obstacle_dist < closest_obstacle_dist:
                    # Check if obstacle is in the direction of the ray
                    obstacle_angle = math.atan2(obstacle.position[1] - self.position[1], 
                                             obstacle.position[0] - self.position[0])
                    angle_diff = abs(angle - obstacle_angle)
                    angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
                    
                    if angle_diff < field_of_view / num_sensors:
                        closest_obstacle_dist = obstacle_dist
                        
                        # Update obstacle value (in sensor index 4)
                        sensor_values = list(sensors[i])
                        sensor_values[4] = 1 - obstacle_dist/vision_range
                        sensors[i] = tuple(sensor_values)
        
        return sensors
        
    def make_decision(self, sensors):
        # Decision making based on sensor input with food type preferences
        turn_strength = 0
        speed_change = 0
        
        # Process each sensor
        for i, (std_food_signal, rich_food_signal, superfood_signal, junk_food_signal, obstacle_signal) in enumerate(sensors):
            sensor_angle = -self.genome['field_of_view']/2 + i * self.genome['field_of_view']/(len(sensors)-1 if len(sensors) > 1 else 1)
            
            # Calculate food attraction with type preferences
            # Standard food
            food_strength = std_food_signal * self.genome['standard_food_pref']
            turn_strength += sensor_angle * food_strength * self.genome['food_attraction']
            speed_change += food_strength * self.genome['food_attraction']
            
            # Rich food (more nutritious)
            food_strength = rich_food_signal * self.genome['rich_food_pref']
            turn_strength += sensor_angle * food_strength * self.genome['food_attraction']
            speed_change += food_strength * self.genome['food_attraction']
            
            # Superfood (extremely nutritious but rare)
            food_strength = superfood_signal * self.genome['superfood_pref']
            turn_strength += sensor_angle * food_strength * self.genome['food_attraction']
            speed_change += food_strength * self.genome['food_attraction']
            
            # Junk food (less nutritious)
            food_strength = junk_food_signal * self.genome['junk_food_pref']
            turn_strength += sensor_angle * food_strength * self.genome['food_attraction']
            speed_change += food_strength * self.genome['food_attraction']
            
            # Obstacle avoidance (steer away)
            turn_strength -= sensor_angle * obstacle_signal * self.genome['obstacle_avoidance']
            speed_change -= obstacle_signal * self.genome['obstacle_avoidance']
            
        # Random exploration factor
        turn_strength += (random.random() - 0.5) * self.genome['exploration_drive']
        
        # Always have a minimum movement
        if abs(speed_change) < 0.1:
            speed_change = 0.5 * self.genome['exploration_drive']
        
        return turn_strength, speed_change
        
    def update(self, foods, obstacles, population):
        # If dead, just update the decay timer
        if not self.alive:
            return
            
        # Increment age
        self.age += 1
        
        # Decrease reproduction cooldown if needed
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1
            
        # Sense the environment
        sensors = self.sense_environment(foods, obstacles)
        
        # Make a decision based on sensory input
        turn, acceleration = self.make_decision(sensors)
        
        # Apply the decision
        self.direction += turn * self.genome['turn_factor']
        self.speed += acceleration * self.genome['speed_factor'] * 0.1
        
        # Always maintain a minimum speed for exploration
        self.speed = max(0.5, min(self.max_speed, self.speed))
        
        # Move the organism
        new_x = self.position[0] + math.cos(self.direction) * self.speed
        new_y = self.position[1] + math.sin(self.direction) * self.speed
        
        # Boundary collision
        if new_x < self.radius:
            new_x = self.radius
            self.direction = math.pi - self.direction
        elif new_x > WINDOW_WIDTH - self.radius:
            new_x = WINDOW_WIDTH - self.radius
            self.direction = math.pi - self.direction
            
        if new_y < self.radius:
            new_y = self.radius
            self.direction = -self.direction
        elif new_y > WINDOW_HEIGHT - self.radius:
            new_y = WINDOW_HEIGHT - self.radius
            self.direction = -self.direction
            
        # Check obstacle collision
        can_move = True
        for obstacle in obstacles[:]:  # Use a copy for safe iteration
            if obstacle.collides_with(new_x, new_y, self.radius):
                can_move = False
                self.direction += math.pi + random.uniform(-0.5, 0.5)  # Bounce with some randomness
                self.energy -= 10  # Energy penalty for hitting obstacle
                
                # Erode the obstacle
                if obstacle.erode():
                    # If obstacle is destroyed, remove it
                    obstacles.remove(obstacle)
                
                break
                
        if can_move:
            self.position = (new_x, new_y)
            
        # Check food collision
        for food in foods:
            if food.active:
                dx = self.position[0] - food.position[0]
                dy = self.position[1] - food.position[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < (self.radius + food.radius):
                    # Add energy from food
                    self.energy += food.energy
                    self.food_eaten += 1
                    
                    # Track which food type was eaten
                    self.food_type_eaten[food.food_type] += 1
                    
                    # Display visual indicator of food type eaten
                    if not FAST_MODE:
                        # Create a small popup text effect showing what was eaten
                        popup_font = pygame.font.SysFont(None, 16)
                        
                        # Determine text based on food type
                        if food.food_type == FoodType.STANDARD:
                            popup_text = "+100"
                        elif food.food_type == FoodType.RICH:
                            popup_text = "+200"
                        elif food.food_type == FoodType.SUPERFOOD:
                            popup_text = "+400!"
                        elif food.food_type == FoodType.JUNK:
                            popup_text = "+50"
                            
                        # Create the popup
                        popup_color = FOOD_COLORS[food.food_type]
                        popup_surf = popup_font.render(popup_text, True, popup_color)
                        
                        # Add the popup to a global list to be drawn and animated
                        # We'll update this during the draw phase
                        popup = {
                            'text': popup_surf,
                            'position': food.position,
                            'lifetime': 30,  # Frames to show
                            'color': popup_color,
                            'y_offset': 0
                        }
                        
                        # Store in a global list to be accessed during draw
                        if not hasattr(Organism, 'food_popups'):
                            Organism.food_popups = []
                        Organism.food_popups.append(popup)
                    
                    # Mark food as consumed
                    food.active = False
                    
                    # Chance to produce poop after eating
                    if random.random() < POOP_CHANCE:
                        # Create a small obstacle (poop) behind the organism
                        behind_angle = self.direction + math.pi  # Opposite direction
                        poop_distance = self.radius * 1.5
                        poop_pos = (
                            self.position[0] + math.cos(behind_angle) * poop_distance,
                            self.position[1] + math.sin(behind_angle) * poop_distance
                        )
                        
                        # Add the new poop to obstacles list
                        obstacles.append(Obstacle(position=poop_pos, poop=True))
        
        # Consume energy (metabolism)
        self.energy -= self.genome['metabolism'] * (1 + self.speed)
        
        # Check for reproduction
        if (self.energy > self.genome['reproduction_threshold'] and 
            self.reproduction_cooldown <= 0 and 
            len(population) < MAX_POPULATION):
            # Create offspring
            offspring_energy = self.energy * self.genome['offspring_energy']
            self.energy -= offspring_energy
            
            # Create offspring with slight offset position
            offset_distance = self.radius * 2
            offset_angle = random.uniform(0, 2 * math.pi)
            offspring_pos = (
                self.position[0] + math.cos(offset_angle) * offset_distance,
                self.position[1] + math.sin(offset_angle) * offset_distance
            )
            
            # Create mutated offspring
            offspring = Organism(
                genome=self.mutate().genome,
                position=offspring_pos,
                parent=self
            )
            offspring.energy = offspring_energy
            
            # Add to population
            population.append(offspring)
            
            # Set reproduction cooldown
            self.reproduction_cooldown = 100  # Frames before can reproduce again
        
        # Check if dead
        if self.energy <= 0:
            self.alive = False
            self.time_of_death = time.time()  # Record time of death for corpse decay
            
    def draw_food_preferences(self, surface, x, y):
        """Draw visual indicators of organism's food preferences."""
        if FAST_MODE:
            return  # Skip in fast mode
            
        # Only draw for living organisms
        if not self.alive:
            return
            
        # Create a small size bar graph showing preferences
        bar_width = 4
        bar_height = 25  # Maximum height
        bar_spacing = 6  # Space between bars
        
        # Draw background bars (light gray)
        bar_bg_height = 25
        for i, food_type in enumerate([FoodType.STANDARD, FoodType.RICH, FoodType.SUPERFOOD, FoodType.JUNK]):
            bar_x = x - (bar_width * 2 + bar_spacing * 1.5) + (i * (bar_width + bar_spacing))
            bar_y = y - self.radius - 40
            pygame.draw.rect(surface, (220, 220, 220), 
                           (bar_x, bar_y, bar_width, bar_bg_height))
        
        # Draw preference bars
        pref_values = [
            self.genome['standard_food_pref'],
            self.genome['rich_food_pref'],
            self.genome['superfood_pref'],
            self.genome['junk_food_pref']
        ]
        
        # Normalize preference values for visualization
        max_pref = max(pref_values)
        
        for i, (food_type, pref) in enumerate(zip(
            [FoodType.STANDARD, FoodType.RICH, FoodType.SUPERFOOD, FoodType.JUNK], 
            pref_values)):
            
            # Normalize height (capped at bar_height)
            norm_height = int((pref / max(max_pref, 1.0)) * bar_height)
            norm_height = min(bar_height, norm_height)
            
            bar_x = x - (bar_width * 2 + bar_spacing * 1.5) + (i * (bar_width + bar_spacing))
            bar_y = y - self.radius - 40 + (bar_height - norm_height)
            
            # Use food colors for the bars
            color = FOOD_COLORS[food_type]
            pygame.draw.rect(surface, color, (bar_x, bar_y, bar_width, norm_height))
            
            # Add small color dot at bottom to identify food type
            pygame.draw.circle(surface, color, 
                             (int(bar_x + bar_width//2), int(y - self.radius - 12)), 
                             2)
    
    def draw(self, surface):
        x, y = int(self.position[0]), int(self.position[1])
        
        if not self.alive:
            # Check if corpse should still be displayed
            if self.time_of_death and time.time() - self.time_of_death > CORPSE_DECAY_TIME / FPS:
                return  # Too old, don't display
                
            # Dead organism fades over time
            if self.time_of_death:
                # Calculate alpha based on time since death
                time_since_death = time.time() - self.time_of_death
                alpha = max(0, 255 - (time_since_death * 255 * FPS / CORPSE_DECAY_TIME))
            else:
                alpha = 150  # Default value if time_of_death not set
                
            # Create transparent surface for drawing dead organism
            death_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            
            if self.decomposing:
                # Draw decomposing organism (turning green/brown)
                # Calculate decomposition progress (0 to 1)
                decomp_progress = min(1.0, self.decompose_timer / CORPSE_TO_FOOD_TIME)
                
                # Color shifts from grey to green/brown as it decomposes
                gray_component = max(20, int(100 * (1 - decomp_progress)))
                green_component = min(200, int(100 + 100 * decomp_progress))
                brown_component = min(100, int(20 + 80 * decomp_progress))
                
                decomp_color = (brown_component, green_component, gray_component, alpha)
                
                # Draw as a blob that's becoming more circular
                if decomp_progress < 0.5:
                    # Still somewhat X-shaped but becoming rounded
                    size = int(self.radius * 0.8)
                    pygame.draw.line(death_surf, decomp_color, 
                                   (x - size, y - size), (x + size, y + size), 
                                   max(1, int(4 * (1 - decomp_progress))))
                    pygame.draw.line(death_surf, decomp_color, 
                                   (x + size, y - size), (x - size, y + size), 
                                   max(1, int(4 * (1 - decomp_progress))))
                    
                    # Add some circular blob elements
                    pygame.draw.circle(death_surf, decomp_color, (x, y), 
                                     int(self.radius * 0.5 * decomp_progress))
                else:
                    # Mostly circular with some bumps for texture
                    pygame.draw.circle(death_surf, decomp_color, (x, y), 
                                     int(self.radius * 0.6))
                    
                    # Add some texture with small circles
                    for i in range(4):
                        angle = random.uniform(0, 2 * math.pi)
                        dist = random.uniform(0.3, 0.8) * self.radius
                        spot_x = x + math.cos(angle) * dist
                        spot_y = y + math.sin(angle) * dist
                        spot_size = random.uniform(1, 3)
                        pygame.draw.circle(death_surf, decomp_color, 
                                         (int(spot_x), int(spot_y)), int(spot_size))
            else:
                # Get color based on species (but grayed out since it's dead)
                if self.species_id and self.species_id in Organism.species_registry:
                    species_info = Organism.species_registry[self.species_id]
                    if 'color' in species_info:
                        base_color = species_info['color']
                        # Make a grayer version of the species color
                        gray_color = (
                            int((base_color[0] + 100) / 2),
                            int((base_color[1] + 100) / 2),
                            int((base_color[2] + 100) / 2),
                            alpha
                        )
                    else:
                        gray_color = (100, 100, 100, alpha)
                else:
                    gray_color = (100, 100, 100, alpha)
                
                # Regular dead organism as an "X"
                size = int(self.radius * 0.8)
                pygame.draw.line(death_surf, gray_color, 
                               (x - size, y - size), (x + size, y + size), 2)
                pygame.draw.line(death_surf, gray_color, 
                               (x + size, y - size), (x - size, y + size), 2)
                
            surface.blit(death_surf, (0, 0))
            return
            
        # Draw body as a small creature with eyes
        
        # Get color based on species
        if self.species_id and self.species_id in Organism.species_registry:
            # Use predefined color for tracked species
            species_info = Organism.species_registry[self.species_id]
            if 'color' in species_info:
                base_color = species_info['color']
            else:
                base_color = BLUE  # Default for untracked species
        else:
            base_color = BLUE  # Default for unidentified species
        
        # Apply individual variation to the species base color
        body_color = (
            max(0, min(255, base_color[0] + self.color_variation[0] // 2)),  # Reduced variation
            max(0, min(255, base_color[1] + self.color_variation[1] // 2)),
            max(0, min(255, base_color[2] + self.color_variation[2] // 2))
        )
        
        # Main body (customized oval based on size factor)
        pygame.draw.ellipse(surface, body_color, 
                          (x - self.radius, y - self.radius*0.8, 
                           self.radius*2, self.radius*1.6))
        
        # Direction as a small head/protrusion
        head_x = x + math.cos(self.direction) * self.radius * 0.8
        head_y = y + math.sin(self.direction) * self.radius * 0.8
        pygame.draw.circle(surface, body_color, (int(head_x), int(head_y)), int(self.radius * 0.5))
        
        # Eyes (two small white circles with black pupils)
        eye_offset = 0.4
        left_eye_angle = self.direction + math.pi/4
        right_eye_angle = self.direction - math.pi/4
        
        left_eye_x = head_x + math.cos(left_eye_angle) * self.radius * eye_offset
        left_eye_y = head_y + math.sin(left_eye_angle) * self.radius * eye_offset
        right_eye_x = head_x + math.cos(right_eye_angle) * self.radius * eye_offset
        right_eye_y = head_y + math.sin(right_eye_angle) * self.radius * eye_offset
        
        # White part of eyes
        pygame.draw.circle(surface, WHITE, (int(left_eye_x), int(left_eye_y)), int(self.radius * 0.2))
        pygame.draw.circle(surface, WHITE, (int(right_eye_x), int(right_eye_y)), int(self.radius * 0.2))
        
        # Pupils (small black dots)
        pygame.draw.circle(surface, BLACK, (int(left_eye_x), int(left_eye_y)), int(self.radius * 0.1))
        pygame.draw.circle(surface, BLACK, (int(right_eye_x), int(right_eye_y)), int(self.radius * 0.1))
        
        # Draw vision field in detailed mode only
        if not FAST_MODE:
            vision_range = self.genome['vision_range']
            field_of_view = self.genome['field_of_view']
            start_angle = self.direction - field_of_view / 2
            end_angle = self.direction + field_of_view / 2
            
            # Create a transparent surface for the vision cone
            vision_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            
            # Draw a transparent triangle instead of an arc (simpler)
            edge1_x = self.position[0] + math.cos(start_angle) * vision_range
            edge1_y = self.position[1] + math.sin(start_angle) * vision_range
            edge2_x = self.position[0] + math.cos(end_angle) * vision_range
            edge2_y = self.position[1] + math.sin(end_angle) * vision_range
            
            # Create a triangle for the vision cone
            vision_points = [
                self.position,
                (int(edge1_x), int(edge1_y)),
                (int(edge2_x), int(edge2_y))
            ]
            pygame.draw.polygon(vision_surf, (255, 255, 0, 15), vision_points)  # More transparent
            
            # Draw the edges of the vision cone (more subtle)
            pygame.draw.line(vision_surf, (255, 255, 0, 30), 
                           (int(self.position[0]), int(self.position[1])), 
                           (int(edge1_x), int(edge1_y)), 1)  # Thinner line
            pygame.draw.line(vision_surf, (255, 255, 0, 30), 
                           (int(self.position[0]), int(self.position[1])), 
                           (int(edge2_x), int(edge2_y)), 1)  # Thinner line
            
            surface.blit(vision_surf, (0, 0))
            
            # Draw food preference indicators
            self.draw_food_preferences(surface, x, y)
        
        # Draw energy bar
        bar_width = 20
        bar_height = 4  # Smaller height
        # Normalize for individual reproduction threshold
        energy_percentage = self.energy / (self.genome['reproduction_threshold'] * 1.2)
        
        # Background for energy bar (gray)
        pygame.draw.rect(surface, GRAY, 
                       (self.position[0] - bar_width/2, 
                        self.position[1] - self.radius - 8,  # Closer to organism
                        bar_width,
                        bar_height))
                        
        # Actual energy level (gradient from red to yellow to green)
        if energy_percentage < 0.3:
            bar_color = RED
        elif energy_percentage < 0.7:
            bar_color = YELLOW
        else:
            bar_color = GREEN
            
        pygame.draw.rect(surface, bar_color, 
                       (self.position[0] - bar_width/2, 
                        self.position[1] - self.radius - 8,
                        bar_width * min(1, max(0, energy_percentage)), 
                        bar_height))
        
        # Draw small generation and species indicator (only in detailed mode)
        if not FAST_MODE:
            small_font = pygame.font.SysFont(None, 14)
            
            # Show generation number
            if self.generation > 1:
                gen_text = str(self.generation)
                gen_surf = small_font.render(gen_text, True, WHITE)
                surface.blit(gen_surf, (x - 3, y - self.radius - 20))
                
            # Show species ID, if we're tracking this species
            if self.species_id and self.species_id in Organism.species_registry:
                species_info = Organism.species_registry[self.species_id]
                if 'rank' in species_info and species_info['rank'] <= MAX_TRACKED_SPECIES:
                    # Use a letter A-F for top species to make it easier to read
                    species_letter = chr(64 + species_info['rank'])  # A=1, B=2, etc.
                    species_surf = small_font.render(species_letter, True, WHITE)
                    surface.blit(species_surf, (x - 3, y - self.radius - 35))
    
    def determine_species(self):
        """Assigns a species ID to this organism based on its genome."""
        # If we're a child, start with our parent's species as default
        if self.parent_species_id is not None:
            parent_genome = None
            if self.parent_species_id in Organism.species_registry:
                parent_genome = Organism.species_registry[self.parent_species_id].get('prototype_genome')
                
            # If we have access to parent species genome, check if we're still similar enough
            if parent_genome:
                # Check genetic distance from parent species
                if self.genetic_distance(parent_genome) <= SPECIES_SIMILARITY_THRESHOLD:
                    # Same species as parent
                    self.species_id = self.parent_species_id
                    return
                    
        # Either no parent species or we've diverged - find a matching species
        for species_id, info in Organism.species_registry.items():
            if 'prototype_genome' in info:
                if self.genetic_distance(info['prototype_genome']) <= SPECIES_SIMILARITY_THRESHOLD:
                    # Found a matching species
                    self.species_id = species_id
                    return
                    
        # No match found - create a new species
        self.species_id = Organism.next_species_id
        
        # Register the new species with this organism's genome as prototype
        Organism.species_registry[self.species_id] = {
            'prototype_genome': self.genome.copy(),
            'creation_time': time.time(),
            'count': 0,
            'peak_count': 0,
            'color': random.choice(SPECIES_COLORS),
            'avg_fitness': 0,
            'max_fitness': 0,
            'rank': 999  # Will be updated during stats collection
        }
        
        Organism.next_species_id += 1
        
    def genetic_distance(self, other_genome):
        """Calculate genetic distance between this organism's genome and another genome."""
        if not other_genome:
            return 1.0  # Maximum distance if no comparison genome
            
        # Calculate normalized distance across all genes
        total_distance = 0
        for gene, value in self.genome.items():
            if gene in other_genome:
                # Normalize by the larger value to get relative difference
                max_val = max(abs(value), abs(other_genome[gene]))
                if max_val == 0:
                    gene_dist = 0  # Both values are 0
                else:
                    gene_dist = abs(value - other_genome[gene]) / max_val
                total_distance += gene_dist
                
        # Average distance across all genes
        return total_distance / len(self.genome)
    
    def mutate(self):
        # Create a copy of the genome
        new_genome = self.genome.copy()
        
        # Mutate each gene
        for gene in new_genome:
            # Determine mutation type
            if random.random() < LARGE_MUTATION_PROBABILITY:
                # Large mutation (±50% change)
                new_genome[gene] *= random.uniform(0.5, 1.5)
            elif random.random() < MUTATION_RATE:
                # Small mutation (±10% change)
                new_genome[gene] *= random.uniform(0.9, 1.1)
                
        # Ensure values stay in reasonable ranges
        new_genome['vision_range'] = max(20, min(300, new_genome['vision_range']))
        new_genome['field_of_view'] = max(math.pi/6, min(math.pi, new_genome['field_of_view']))
        new_genome['num_sensors'] = max(3, min(9, new_genome['num_sensors']))
        new_genome['turn_factor'] = max(0.05, min(1.0, new_genome['turn_factor']))
        new_genome['speed_factor'] = max(0.2, min(3.0, new_genome['speed_factor']))
        new_genome['metabolism'] = max(0.05, min(0.5, new_genome['metabolism']))
        
        # Food preferences - allow specialization but maintain reasonable ranges
        new_genome['standard_food_pref'] = max(0.1, min(3.0, new_genome['standard_food_pref']))
        new_genome['rich_food_pref'] = max(0.1, min(3.0, new_genome['rich_food_pref']))
        new_genome['superfood_pref'] = max(0.1, min(3.0, new_genome['superfood_pref']))
        new_genome['junk_food_pref'] = max(0.1, min(3.0, new_genome['junk_food_pref']))
        
        # Reproduction traits
        new_genome['reproduction_threshold'] = max(600, min(1200, new_genome['reproduction_threshold']))
        new_genome['offspring_energy'] = max(0.1, min(0.5, new_genome['offspring_energy']))
        new_genome['size_factor'] = max(0.6, min(1.5, new_genome['size_factor']))
        
        # Create a new organism with the mutated genome
        # Note: Position and parent will be set properly when reproduction actually happens
        return Organism(new_genome)

def calculate_fitness(organism):
    # The fitness function considers:
    # 1. How much food was eaten (with bonuses for high-energy food)
    # 2. How much energy remains
    # 3. How long the organism survived
    # 4. Reproductive success
    
    # Base fitness from raw food count
    base_fitness = organism.food_eaten * 50
    
    # Add bonuses for nutritious food
    # Standard food is already counted in base
    food_bonus = (
        organism.food_type_eaten[FoodType.RICH] * 75 +        # Bonus for rich food
        organism.food_type_eaten[FoodType.SUPERFOOD] * 150 +  # Major bonus for superfood
        organism.food_type_eaten[FoodType.JUNK] * 10          # Small bonus for junk food
    )
    
    # Calculate overall fitness
    fitness = base_fitness + food_bonus
    
    # Bonuses for living organisms
    if organism.alive:
        fitness += organism.energy * 0.1  # Bonus for remaining energy
        fitness += organism.age * 0.01    # Small bonus for longevity
    
    return fitness

def evolve_population(population):
    # Calculate fitness for all organisms
    for organism in population:
        organism.fitness = calculate_fitness(organism)
        
    # Sort by fitness
    population.sort(key=lambda x: x.fitness, reverse=True)
    
    # Keep the best 20% of the population
    elites = population[:len(population) // 5]
    
    # Create new organisms through mutation
    new_population = []
    while len(new_population) < POPULATION_SIZE:
        # Select parent weighted by fitness
        total_fitness = sum(o.fitness for o in elites)
        if total_fitness == 0:
            # If all have zero fitness, select randomly
            parent = random.choice(elites)
        else:
            # Weighted selection
            selection_point = random.uniform(0, total_fitness)
            current_sum = 0
            for organism in elites:
                current_sum += organism.fitness
                if current_sum >= selection_point:
                    parent = organism
                    break
            else:
                parent = elites[-1]  # Fallback
        
        # Add a mutated copy to the new population
        new_population.append(parent.mutate())
        
    return new_population

def simulate_generation(population, screen):
    # Setup environment
    foods = [Food() for _ in range(FOOD_COUNT)]
    obstacles = [Obstacle() for _ in range(OBSTACLE_COUNT)]
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    
    # Track statistics
    best_score = 0
    avg_score = 0
    
    # Run simulation for specified number of steps
    for step in range(SIMULATION_STEPS):
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Toggle fast mode with spacebar
                    global FAST_MODE
                    FAST_MODE = not FAST_MODE
                    print(f"Fast mode: {FAST_MODE}")
        
        # Update organisms
        for organism in population:
            organism.update(foods, obstacles)
        
        # Count active organisms and foods
        alive_count = sum(1 for organism in population if organism.alive)
        food_count = sum(1 for food in foods if food.active)
        
        # End early if all organisms die or all food is eaten
        if alive_count == 0 or food_count == 0:
            break
        
        # In fast mode, only draw every 5 steps
        if not FAST_MODE or step % 5 == 0:
            # Draw everything
            screen.fill(WHITE)
            
            # Draw obstacles and food
            for obstacle in obstacles:
                obstacle.draw(screen)
            for food in foods:
                food.draw(screen)
                
            # Draw organisms - in fast mode, simplify the drawing
            for organism in population:
                if FAST_MODE:
                    # Simplified drawing for fast mode
                    if organism.alive:
                        pygame.draw.circle(surface=screen, 
                                         color=BLUE, 
                                         center=(int(organism.position[0]), int(organism.position[1])), 
                                         radius=organism.radius)
                else:
                    organism.draw(screen)
                
            # Display statistics
            current_avg = sum(o.food_eaten for o in population) / len(population)
            current_best = max(o.food_eaten for o in population)
            best_score = max(best_score, current_best)
            avg_score = current_avg
            
            status_text = f"Step: {step}/{SIMULATION_STEPS} | Alive: {alive_count}/{POPULATION_SIZE} | Food: {food_count}/{FOOD_COUNT}"
            score_text = f"Avg Food: {avg_score:.2f} | Best Food: {best_score}"
            mode_text = f"{'FAST MODE' if FAST_MODE else 'DETAILED MODE'} - Press SPACE to toggle"
            
            status_surface = font.render(status_text, True, BLACK)
            score_surface = font.render(score_text, True, BLACK)
            mode_surface = font.render(mode_text, True, RED if FAST_MODE else GREEN)
            
            screen.blit(status_surface, (10, 10))
            screen.blit(score_surface, (10, 40))
            screen.blit(mode_surface, (10, 70))
            
            # Update display
            pygame.display.flip()
            
        # Maintain framerate - faster in fast mode
        clock.tick(FPS * (2 if FAST_MODE else 1))
    
    # End of generation - calculate and return statistics
    return {
        'best_score': best_score,
        'avg_score': avg_score,
        'alive_count': alive_count
    }

def update_species_statistics(population):
    """Update species statistics and rank the top species."""
    # Reset counts for all species
    for species_id in Organism.species_registry:
        Organism.species_registry[species_id]['count'] = 0
        Organism.species_registry[species_id]['avg_fitness'] = 0
        
    # Count organisms per species and calculate fitness
    species_organisms = {}
    for organism in population:
        if organism.species_id:
            if organism.species_id not in species_organisms:
                species_organisms[organism.species_id] = []
            species_organisms[organism.species_id].append(organism)
    
    # Update species stats
    for species_id, organisms in species_organisms.items():
        if species_id in Organism.species_registry:
            species_info = Organism.species_registry[species_id]
            count = len(organisms)
            species_info['count'] = count
            species_info['peak_count'] = max(species_info['peak_count'], count)
            
            # Calculate fitness stats
            if organisms:
                avg_fitness = sum(calculate_fitness(o) for o in organisms) / len(organisms)
                max_fitness = max(calculate_fitness(o) for o in organisms)
                species_info['avg_fitness'] = avg_fitness
                species_info['max_fitness'] = max(species_info['max_fitness'], max_fitness)
                
                # Calculate food preferences
                avg_prefs = {
                    'standard_food_pref': sum(o.genome['standard_food_pref'] for o in organisms) / len(organisms),
                    'rich_food_pref': sum(o.genome['rich_food_pref'] for o in organisms) / len(organisms),
                    'superfood_pref': sum(o.genome['superfood_pref'] for o in organisms) / len(organisms),
                    'junk_food_pref': sum(o.genome['junk_food_pref'] for o in organisms) / len(organisms)
                }
                species_info['food_preferences'] = avg_prefs
                
                # Determine food specialization (if any)
                # A species is specialized if one food preference is significantly higher than others
                pref_values = list(avg_prefs.values())
                max_pref = max(pref_values)
                avg_other_prefs = sum(p for p in pref_values if p != max_pref) / 3  # Average of other preferences
                
                # If max preference is at least 50% higher than average of others, consider specialized
                if max_pref > avg_other_prefs * 1.5:
                    # Find which food type is specialized
                    for food_type, pref_name in [
                        (FoodType.STANDARD, 'standard_food_pref'),
                        (FoodType.RICH, 'rich_food_pref'),
                        (FoodType.SUPERFOOD, 'superfood_pref'),
                        (FoodType.JUNK, 'junk_food_pref')
                    ]:
                        if avg_prefs[pref_name] == max_pref:
                            species_info['specialized_food'] = food_type
                            break
                else:
                    species_info['specialized_food'] = None  # Not specialized
    
    # Rank species by population count (could also use fitness)
    active_species = [(sid, info) for sid, info in Organism.species_registry.items() 
                     if info['count'] > 0]
    ranked_species = sorted(active_species, key=lambda x: x[1]['count'], reverse=True)
    
    # Update ranks (only for active species)
    for rank, (species_id, _) in enumerate(ranked_species, 1):
        Organism.species_registry[species_id]['rank'] = rank

def simulate_continuous_evolution(screen):
    # Setup environment with initial population
    population = [Organism() for _ in range(POPULATION_SIZE)]
    
    # Create initial food with different types based on probabilities
    foods = []
    for _ in range(FOOD_COUNT):
        foods.append(Food())  # Food constructor handles random type assignment
        
    obstacles = [Obstacle() for _ in range(OBSTACLE_COUNT)]
    
    # Initialize utilities
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    
    # Track statistics
    step = 0
    food_respawn_timer = 0
    highest_generation = 1
    stats_history = []  # Store stats over time
    stat_collection_interval = 500  # Collect stats every 500 steps
    
    # Initialize flags and controls
    running = True
    paused = False
    
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Toggle fast mode
                    global FAST_MODE
                    FAST_MODE = not FAST_MODE
                    print(f"Fast mode: {FAST_MODE}")
                elif event.key == pygame.K_p:
                    # Toggle pause
                    paused = not paused
                    print(f"Paused: {paused}")
                    
                # Food type probability adjustments
                elif event.key == pygame.K_1:
                    # Increase standard food probability
                    FOOD_TYPE_PROBS[FoodType.STANDARD] = min(0.9, FOOD_TYPE_PROBS[FoodType.STANDARD] + 0.1)
                    # Normalize probabilities
                    total = sum(FOOD_TYPE_PROBS.values())
                    for food_type in FOOD_TYPE_PROBS:
                        if food_type != FoodType.STANDARD:
                            FOOD_TYPE_PROBS[food_type] *= (1.0 - FOOD_TYPE_PROBS[FoodType.STANDARD]) / (total - FOOD_TYPE_PROBS[FoodType.STANDARD])
                    print(f"Food type probabilities: {FOOD_TYPE_PROBS}")
                    
                elif event.key == pygame.K_2:
                    # Increase rich food probability
                    FOOD_TYPE_PROBS[FoodType.RICH] = min(0.9, FOOD_TYPE_PROBS[FoodType.RICH] + 0.1)
                    # Normalize probabilities
                    total = sum(FOOD_TYPE_PROBS.values())
                    for food_type in FOOD_TYPE_PROBS:
                        if food_type != FoodType.RICH:
                            FOOD_TYPE_PROBS[food_type] *= (1.0 - FOOD_TYPE_PROBS[FoodType.RICH]) / (total - FOOD_TYPE_PROBS[FoodType.RICH])
                    print(f"Food type probabilities: {FOOD_TYPE_PROBS}")
                    
                elif event.key == pygame.K_3:
                    # Increase superfood probability
                    FOOD_TYPE_PROBS[FoodType.SUPERFOOD] = min(0.5, FOOD_TYPE_PROBS[FoodType.SUPERFOOD] + 0.05)
                    # Normalize probabilities
                    total = sum(FOOD_TYPE_PROBS.values())
                    for food_type in FOOD_TYPE_PROBS:
                        if food_type != FoodType.SUPERFOOD:
                            FOOD_TYPE_PROBS[food_type] *= (1.0 - FOOD_TYPE_PROBS[FoodType.SUPERFOOD]) / (total - FOOD_TYPE_PROBS[FoodType.SUPERFOOD])
                    print(f"Food type probabilities: {FOOD_TYPE_PROBS}")
                    
                elif event.key == pygame.K_4:
                    # Increase junk food probability
                    FOOD_TYPE_PROBS[FoodType.JUNK] = min(0.9, FOOD_TYPE_PROBS[FoodType.JUNK] + 0.1)
                    # Normalize probabilities
                    total = sum(FOOD_TYPE_PROBS.values())
                    for food_type in FOOD_TYPE_PROBS:
                        if food_type != FoodType.JUNK:
                            FOOD_TYPE_PROBS[food_type] *= (1.0 - FOOD_TYPE_PROBS[FoodType.JUNK]) / (total - FOOD_TYPE_PROBS[FoodType.JUNK])
                    print(f"Food type probabilities: {FOOD_TYPE_PROBS}")
                    
                elif event.key == pygame.K_r:
                    # Reset food probabilities to default
                    FOOD_TYPE_PROBS[FoodType.STANDARD] = 0.6
                    FOOD_TYPE_PROBS[FoodType.RICH] = 0.25
                    FOOD_TYPE_PROBS[FoodType.SUPERFOOD] = 0.05
                    FOOD_TYPE_PROBS[FoodType.JUNK] = 0.1
                    print(f"Food type probabilities reset to default: {FOOD_TYPE_PROBS}")
        
        # Skip simulation if paused
        if paused:
            # Still render but don't update
            draw_simulation(screen, population, foods, obstacles, step, font)
            clock.tick(FPS // 4)  # Slow down while paused
            continue
            
        # Update step counter
        step += 1
        
        # Update organisms and handle corpse decomposition
        alive_count = 0
        for organism in population[:]:  # Create a copy of the list for safe iteration
            if organism.alive:
                organism.update(foods, obstacles, population)
                alive_count += 1
            else:
                # Handle dead organisms
                if organism.time_of_death:
                    time_since_death = time.time() - organism.time_of_death
                    
                    # Start decomposition after a delay
                    if not organism.decomposing and time_since_death > CORPSE_DECAY_TIME / (FPS * 3):
                        organism.decomposing = True
                        
                    # Update decomposition state
                    if organism.decomposing:
                        organism.decompose_timer += 1
                        
                        # When fully decomposed, turn into food and remove organism
                        if organism.decompose_timer >= CORPSE_TO_FOOD_TIME:
                            # Determine food type based on organism traits
                            # Larger organisms tend to become richer food
                            # Organisms that have eaten superfood have a chance to become superfood
                            
                            # Calculate probabilities based on organism traits
                            size_factor = organism.genome['size_factor']
                            metabolism = organism.genome['metabolism']
                            
                            # Track what the organism ate in its lifetime
                            superfood_ratio = organism.food_type_eaten.get(FoodType.SUPERFOOD, 0) / max(1, sum(organism.food_type_eaten.values()))
                            rich_food_ratio = organism.food_type_eaten.get(FoodType.RICH, 0) / max(1, sum(organism.food_type_eaten.values()))
                            
                            # Base probabilities
                            type_probs = {
                                FoodType.STANDARD: 0.6,
                                FoodType.RICH: 0.25,
                                FoodType.SUPERFOOD: 0.05,
                                FoodType.JUNK: 0.1
                            }
                            
                            # Modify based on organism traits
                            # Larger organisms tend to become richer food
                            if size_factor > 1.1:
                                type_probs[FoodType.RICH] += 0.15
                                type_probs[FoodType.STANDARD] -= 0.1
                                type_probs[FoodType.JUNK] -= 0.05
                            
                            # High metabolism organisms might become junk food (used up resources)
                            if metabolism > 0.25:
                                type_probs[FoodType.JUNK] += 0.1
                                type_probs[FoodType.RICH] -= 0.05
                                type_probs[FoodType.STANDARD] -= 0.05
                            
                            # Organisms that ate superfood might become superfood
                            if superfood_ratio > 0.1:
                                type_probs[FoodType.SUPERFOOD] += superfood_ratio * 0.3
                                type_probs[FoodType.STANDARD] -= superfood_ratio * 0.3
                            
                            # Organisms that ate rich food might become rich food
                            if rich_food_ratio > 0.2:
                                type_probs[FoodType.RICH] += rich_food_ratio * 0.2
                                type_probs[FoodType.STANDARD] -= rich_food_ratio * 0.2
                            
                            # Normalize probabilities
                            total = sum(type_probs.values())
                            for food_type in type_probs:
                                type_probs[food_type] /= total
                            
                            # Select food type based on probabilities
                            r = random.random()
                            cumulative = 0
                            selected_type = FoodType.STANDARD  # Default
                            for food_type, prob in type_probs.items():
                                cumulative += prob
                                if r <= cumulative:
                                    selected_type = food_type
                                    break
                            
                            # Create food at the organism's position with selected type
                            new_food = Food(position=organism.position, food_type=selected_type)
                            
                            # Make corpse-derived food slightly more nutritious
                            new_food.energy *= 1.2
                            new_food.age = 100  # Start somewhat "ripened"
                            
                            foods.append(new_food)
                            
                            # Remove the decomposed organism
                            population.remove(organism)
                    
                    # Remove very old corpses that haven't decomposed for some reason
                    elif time_since_death > CORPSE_DECAY_TIME / FPS:
                        population.remove(organism)
        
        # Update food and obstacles
        for food in foods:
            food.update()
            
        for obstacle in obstacles:
            obstacle.update()
            
        # Regenerate obstacles occasionally if too few remain
        if len(obstacles) < OBSTACLE_COUNT * 0.6 and random.random() < 0.01:
            obstacles.append(Obstacle())
        
        # Regenerate food occasionally
        food_respawn_timer += 1
        active_food_count = sum(1 for food in foods if food.active)
        
        # Ensure there's always at least a minimum amount of food
        if active_food_count < FOOD_COUNT * 0.2 or random.random() < FOOD_RESPAWN_RATE:
            # Find an inactive food to reactivate, or create a new one if all are active
            inactive_foods = [f for f in foods if not f.active]
            if inactive_foods:
                food = random.choice(inactive_foods)
                food.position = (
                    random.randint(50, WINDOW_WIDTH - 50),
                    random.randint(50, WINDOW_HEIGHT - 50)
                )
                food.active = True
                food.energy = 100
                food.age = 0
            else:
                # All food is active, potentially add a new one if we're below limit
                if len(foods) < FOOD_COUNT * 2:
                    # Ensure we maintain reasonable food type distribution
                    # Add food with weighted probabilities based on current scarcity
                    
                    # Count each type of active food to calculate scarcity
                    food_counts = {
                        FoodType.STANDARD: 0,
                        FoodType.RICH: 0, 
                        FoodType.SUPERFOOD: 0,
                        FoodType.JUNK: 0
                    }
                    
                    for food in foods:
                        if food.active:
                            food_counts[food.food_type] += 1
                    
                    # Calculate food type scarcity (invert counts to make scarce types more likely)
                    type_weights = {}
                    for food_type in [FoodType.STANDARD, FoodType.RICH, FoodType.SUPERFOOD, FoodType.JUNK]:
                        # Base probability from global settings
                        base_prob = FOOD_TYPE_PROBS[food_type]
                        
                        # Adjust based on current scarcity (fewer = higher chance)
                        scarcity_factor = 1.0 - (food_counts[food_type] / max(1, active_food_count))
                        
                        # Calculate final weight, ensuring we maintain some baseline probability
                        type_weights[food_type] = base_prob * (0.5 + 0.5 * scarcity_factor)
                    
                    # Normalize weights to sum to 1.0
                    total_weight = sum(type_weights.values())
                    if total_weight > 0:
                        for food_type in type_weights:
                            type_weights[food_type] /= total_weight
                    
                    # Select food type based on adjusted weights
                    r = random.random()
                    cumulative = 0
                    selected_type = FoodType.STANDARD  # Default
                    for food_type, weight in type_weights.items():
                        cumulative += weight
                        if r <= cumulative:
                            selected_type = food_type
                            break
                    
                    # Add the new food
                    foods.append(Food(food_type=selected_type))
        
        # Track statistics
        if step % stat_collection_interval == 0:
            if population:
                # Update species statistics
                update_species_statistics(population)
                
                # Get generation stats
                avg_generation = sum(o.generation for o in population) / len(population)
                max_generation = max(o.generation for o in population)
                highest_generation = max(highest_generation, max_generation)
                
                # Count species
                species_count = len([s for s, info in Organism.species_registry.items() 
                                   if info['count'] > 0])
                
                # Get top species info if any exists
                top_species_info = None
                top_species = [s for s, info in Organism.species_registry.items() 
                              if info['rank'] == 1 and info['count'] > 0]
                if top_species:
                    top_species_id = top_species[0]
                    top_species_info = Organism.species_registry[top_species_id]
                
                stats = {
                    'step': step,
                    'population': len(population),
                    'alive_count': alive_count,
                    'food_count': active_food_count,
                    'avg_generation': avg_generation,
                    'max_generation': max_generation,
                    'species_count': species_count,
                    'top_species_id': top_species[0] if top_species else None,
                    'top_species_count': top_species_info['count'] if top_species_info else 0,
                    'top_species_fitness': top_species_info['avg_fitness'] if top_species_info else 0,
                }
                stats_history.append(stats)
                
                # Keep only last 1000 stat points to prevent memory bloat
                if len(stats_history) > 1000:
                    stats_history.pop(0)
        
        # Draw everything
        draw_simulation(screen, population, foods, obstacles, step, font)
        
        # Maintain framerate
        clock.tick(FPS * (3 if FAST_MODE else 1))
    
    return stats_history

def draw_simulation(screen, population, foods, obstacles, step, font):
    # Only draw every few steps in fast mode to improve performance
    if FAST_MODE and step % 3 != 0:
        return
        
    # Clear screen
    screen.fill(WHITE)
    
    # Draw obstacles and food
    for obstacle in obstacles:
        obstacle.draw(screen)
        
    for food in foods:
        food.draw(screen)
        
    # Draw organisms
    alive_organisms = [o for o in population if o.alive]
    dead_organisms = [o for o in population if not o.alive]
    
    # Draw dead organisms first (so alive ones appear on top)
    for organism in dead_organisms:
        organism.draw(screen)
        
    # Then draw alive ones
    for organism in alive_organisms:
        organism.draw(screen)
        
    # Draw food consumption popups if any
    if hasattr(Organism, 'food_popups') and not FAST_MODE:
        popups_to_keep = []
        
        for popup in Organism.food_popups:
            # Update popup
            popup['lifetime'] -= 1
            popup['y_offset'] -= 0.5  # Float upward
            
            # Draw popup with fade out effect
            alpha = min(255, int(255 * popup['lifetime'] / 30))
            
            # Create a surface with alpha
            popup_alpha_surf = pygame.Surface((popup['text'].get_width(), popup['text'].get_height()), 
                                            pygame.SRCALPHA)
            popup_alpha_surf.fill((0, 0, 0, 0))  # Transparent
            
            # Blit text onto alpha surface with fading
            popup_alpha_surf.blit(popup['text'], (0, 0))
            popup_alpha_surf.set_alpha(alpha)
            
            # Draw to screen
            screen.blit(popup_alpha_surf, 
                       (popup['position'][0] - popup['text'].get_width()//2, 
                        popup['position'][1] - 15 + int(popup['y_offset'])))
            
            # Keep if still alive
            if popup['lifetime'] > 0:
                popups_to_keep.append(popup)
                
        # Update the list
        Organism.food_popups = popups_to_keep
    
    # Calculate statistics
    alive_count = len(alive_organisms)
    total_count = len(population)
    
    # Count each type of active food
    food_counts = {
        FoodType.STANDARD: 0,
        FoodType.RICH: 0,
        FoodType.SUPERFOOD: 0,
        FoodType.JUNK: 0
    }
    
    for food in foods:
        if food.active:
            food_counts[food.food_type] += 1
            
    active_food = sum(food_counts.values())
    
    max_gen = max((o.generation for o in population), default=1)
    avg_gen = sum(o.generation for o in population) / max(1, len(population))
    
    # Count poop obstacles for stats
    poop_count = sum(1 for o in obstacles if o.is_poop)
    
    # Count species for stats
    species_count = len([s for s, info in Organism.species_registry.items() 
                       if info['count'] > 0])
                       
    # Get top species info
    top_species = [s for s, info in Organism.species_registry.items() 
                  if info['rank'] == 1 and info['count'] > 0]
    top_species_id = top_species[0] if top_species else None
    
    # Display statistics
    stats_text = [
        f"Step: {step} | Population: {alive_count}/{total_count} | Food: {active_food}/{len(foods)}",
        f"Generations: Avg {avg_gen:.1f} | Max {max_gen} | Species: {species_count}",
        f"Obstacles: {len(obstacles) - poop_count}/{poop_count} | {'FAST MODE' if FAST_MODE else 'DETAILED MODE'} | P: pause",
        f"Food Controls: 1-4 adjust food types, R: reset distribution"
    ]
    
    # Add food type distribution
    if not FAST_MODE:
        # Draw food type legend
        legend_y = 75
        legend_text = "Food Types: "
        legend_surf = font.render(legend_text, True, BLACK)
        screen.blit(legend_surf, (10, legend_y))
        
        # Draw colored circles for each food type
        circle_size = 8
        x_offset = 10 + legend_surf.get_width() + 5
        
        # Display all food types with consistent formatting
        for food_type in [FoodType.STANDARD, FoodType.RICH, FoodType.SUPERFOOD, FoodType.JUNK]:
            color = FOOD_COLORS[food_type]
            pygame.draw.circle(screen, color, (x_offset, legend_y + circle_size), circle_size)
            
            # Add a small glow effect for superfood and rich food to match their appearance
            if food_type == FoodType.SUPERFOOD or food_type == FoodType.RICH:
                glow_radius = circle_size + 2
                glow_color = (color[0], color[1], color[2], 100)  # Semi-transparent
                glow_surf = pygame.Surface((40, 40), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, glow_color, (circle_size, circle_size), glow_radius)
                screen.blit(glow_surf, (x_offset - circle_size, legend_y))
            
            # Display count with food type name
            count_text = f"{FOOD_TYPE_NAMES[food_type]}: {food_counts[food_type]}"
            
            # Add nutritional value indicator
            if food_type == FoodType.STANDARD:
                count_text += " (100)"
            elif food_type == FoodType.RICH:
                count_text += " (200)"
            elif food_type == FoodType.SUPERFOOD:
                count_text += " (400)"
            elif food_type == FoodType.JUNK:
                count_text += " (50)"
                
            food_surf = font.render(count_text, True, BLACK)
            screen.blit(food_surf, (x_offset + circle_size + 5, legend_y))
            x_offset += food_surf.get_width() + 20
    
    # Add species key if we have tracked species and in detailed mode
    if not FAST_MODE and species_count > 0:
        # Draw species legend
        legend_y = 100  # Below food type legend
        legend_text = "Species: "
        legend_surf = font.render(legend_text, True, BLACK)
        screen.blit(legend_surf, (10, legend_y))
        
        # Draw colored squares for top species
        square_size = 15
        x_offset = 10 + legend_surf.get_width() + 5
        
        top_species_to_show = min(MAX_TRACKED_SPECIES, species_count)
        for rank in range(1, top_species_to_show + 1):
            matching_species = [s for s, info in Organism.species_registry.items() 
                              if info['rank'] == rank and info['count'] > 0]
            if matching_species:
                species_id = matching_species[0]
                species_info = Organism.species_registry[species_id]
                
                # Draw colored square
                color = species_info['color']
                pygame.draw.rect(screen, color, 
                               (x_offset, legend_y, square_size, square_size))
                
                # Add letter label
                species_letter = chr(64 + rank)  # A=1, B=2, etc.
                label_surf = font.render(species_letter, True, BLACK)
                screen.blit(label_surf, (x_offset + square_size + 2, legend_y))
                
                # Add count
                count_surf = font.render(f"{species_info['count']}", True, BLACK)
                screen.blit(count_surf, (x_offset + square_size + 15, legend_y))
                
                # Show food specialization if any
                if 'specialized_food' in species_info and species_info['specialized_food'] is not None:
                    food_type = species_info['specialized_food']
                    # Add a small indicator of the food type specialized in
                    food_color = FOOD_COLORS[food_type]
                    pygame.draw.circle(screen, food_color, 
                                     (x_offset + square_size + 35, legend_y + square_size//2), 4)
                    
                    # Add a small "S" for "specialized"
                    spec_surf = font.render("S", True, BLACK)
                    screen.blit(spec_surf, (x_offset + square_size + 40, legend_y))
                
                x_offset += square_size + 40
    
    for i, text in enumerate(stats_text):
        text_surface = font.render(text, True, BLACK)
        screen.blit(text_surface, (10, 10 + i * 25))
    
    # Update display
    pygame.display.flip()

def main():
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Persistent Organism Evolution")
    
    # Run the continuous simulation
    stats = simulate_continuous_evolution(screen)
    
    # When simulation ends, show final stats graph if we have enough data
    if len(stats) > 1:
        # Display final stats graph
        screen.fill(WHITE)
        font = pygame.font.SysFont(None, 24)
        
        # Create a line graph of performance over time
        graph_width = 700
        graph_height = 400
        graph_margin = 50
        
        # Draw graph background
        pygame.draw.rect(screen, (240, 240, 240), 
                       (graph_margin, graph_margin, 
                        graph_width, graph_height))
        
        # Draw axes
        pygame.draw.line(screen, BLACK, 
                       (graph_margin, graph_margin + graph_height),
                       (graph_margin + graph_width, graph_margin + graph_height), 2)
        pygame.draw.line(screen, BLACK,
                       (graph_margin, graph_margin + graph_height),
                       (graph_margin, graph_margin), 2)
                      
        # Find max values for scaling
        max_population = max(s['population'] for s in stats)
        max_generation = max(s['max_generation'] for s in stats)
        max_steps = stats[-1]['step']
        
        # Track species of interest
        # Extract unique species IDs that were "top species" at some point
        dominant_species = set()
        for stat in stats:
            if 'top_species_id' in stat and stat['top_species_id'] is not None:
                dominant_species.add(stat['top_species_id'])
        
        # Prepare species legend and colors
        species_colors = {}
        for i, species_id in enumerate(dominant_species):
            if species_id in Organism.species_registry:
                species_colors[species_id] = Organism.species_registry[species_id].get('color', (0, 0, 0))
            else:
                species_colors[species_id] = SPECIES_COLORS[i % len(SPECIES_COLORS)]
        
        # Draw data points and lines
        pop_points = []  # Total population
        gen_points = []  # Max generation
        species_points = {}  # Points per dominant species
        
        for i, stat in enumerate(stats):
            x = graph_margin + (stat['step'] / max_steps) * graph_width
            
            # Population (blue)
            pop_y = graph_margin + graph_height - (stat['population'] / max_population * graph_height * 0.8)
            pop_points.append((x, pop_y))
            
            # Maximum generation (red)
            gen_y = graph_margin + graph_height - (stat['max_generation'] / max_generation * graph_height * 0.8)
            gen_points.append((x, gen_y))
            
            # Species counts
            if 'top_species_id' in stat and stat['top_species_id'] is not None:
                species_id = stat['top_species_id']
                if species_id not in species_points:
                    species_points[species_id] = []
                
                if 'top_species_count' in stat and stat['top_species_count'] > 0:
                    # Plot species population normalized to max population
                    species_y = graph_margin + graph_height - (stat['top_species_count'] / max_population * graph_height * 0.8)
                    species_points[species_id].append((x, species_y))
            
        # Draw species lines first (under the main metrics)
        for species_id, points in species_points.items():
            if len(points) > 1:
                color = species_colors.get(species_id, (100, 100, 100))
                pygame.draw.lines(screen, color, False, points, 2)
        
        # Draw main metric lines
        if len(pop_points) > 1:
            pygame.draw.lines(screen, (0, 0, 255), False, pop_points, 2)
            pygame.draw.lines(screen, (255, 0, 0), False, gen_points, 2)
            
        # Draw axis labels
        title = font.render("Evolution Progress", True, BLACK)
        x_label = font.render("Simulation Steps", True, BLACK)
        y_label = font.render("Value", True, BLACK)
        
        screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 10))
        screen.blit(x_label, (WINDOW_WIDTH // 2 - x_label.get_width() // 2, 
                            graph_margin + graph_height + 10))
        
        # Rotate Y label
        y_label_rotated = pygame.transform.rotate(y_label, 90)
        screen.blit(y_label_rotated, (graph_margin - 40, 
                                    graph_margin + graph_height // 2 - y_label_rotated.get_height() // 2))
                                    
        # Draw main metrics legend
        legend_y = graph_margin + 20
        pygame.draw.line(screen, (255, 0, 0), (graph_margin + graph_width + 10, legend_y),
                       (graph_margin + graph_width + 40, legend_y), 2)
        gen_label = font.render("Max Generation", True, BLACK)
        screen.blit(gen_label, (graph_margin + graph_width + 50, legend_y - 5))
        
        legend_y += 30
        pygame.draw.line(screen, (0, 0, 255), (graph_margin + graph_width + 10, legend_y),
                       (graph_margin + graph_width + 40, legend_y), 2)
        pop_label = font.render("Total Population", True, BLACK)
        screen.blit(pop_label, (graph_margin + graph_width + 50, legend_y - 5))
        
        # Draw species legend (showing top species that were dominant at some point)
        legend_y += 40
        species_title = font.render("Dominant Species:", True, BLACK)
        screen.blit(species_title, (graph_margin + graph_width + 10, legend_y))
        
        for i, (species_id, color) in enumerate(species_colors.items()):
            if i < 6:  # Limit to 6 species in legend to avoid clutter
                legend_y += 20
                pygame.draw.line(screen, color, (graph_margin + graph_width + 10, legend_y),
                               (graph_margin + graph_width + 40, legend_y), 2)
                
                # Show species ID and any info we have
                if species_id in Organism.species_registry:
                    info = Organism.species_registry[species_id]
                    species_label = font.render(f"Species {species_id} (Gen {info.get('rank', '?')})", True, BLACK)
                else:
                    species_label = font.render(f"Species {species_id}", True, BLACK)
                    
                screen.blit(species_label, (graph_margin + graph_width + 50, legend_y - 5))
        
        pygame.display.flip()
        
        # Wait for quit
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    waiting = False
    
    pygame.quit()

if __name__ == "__main__":
    main()