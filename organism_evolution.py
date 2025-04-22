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

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 200, 0)  # Darker green for food
BLUE = (50, 100, 255)  # Lighter blue for organisms
YELLOW = (255, 255, 0)
GRAY = (80, 80, 80)  # Darker gray for obstacles
BROWN = (139, 69, 19)  # Brown for obstacles

class Food:
    def __init__(self, position=None):
        if position:
            self.position = position
        else:
            self.position = (
                random.randint(50, WINDOW_WIDTH - 50),
                random.randint(50, WINDOW_HEIGHT - 50)
            )
        self.radius = 5
        self.energy = 100
        self.active = True
        self.age = 0
        
    def update(self):
        # Food becomes more nutritious as it ages (simulating growth)
        if self.active and self.age < 300:  # Cap at 300 frames
            self.age += 1
            if self.age % 50 == 0 and self.energy < 200:
                self.energy += 10
            
    def draw(self, surface):
        if self.active:
            # Draw food as a bright green apple-like shape
            # Bigger and darker green as it ages/becomes more nutritious
            size_factor = min(1.5, 1.0 + self.age/300)
            radius = int(self.radius * size_factor)
            
            # Calculate color based on age (gets darker/riper)
            green_val = max(100, 200 - self.age/3)
            color = (0, green_val, 0)
            
            pygame.draw.circle(surface, color, self.position, radius)
            # Add a small stem
            pygame.draw.line(surface, BROWN, 
                          (self.position[0], self.position[1] - radius),
                          (self.position[0] + 3, self.position[1] - radius - 3), 2)

class Obstacle:
    def __init__(self):
        self.position = (
            random.randint(50, WINDOW_WIDTH - 50),
            random.randint(50, WINDOW_HEIGHT - 50)
        )
        self.radius = random.randint(20, 40)
        self.direction = random.uniform(0, 2 * math.pi)
        self.speed = 0 
        self.stationary = random.random() < 0.6  # 60% chance to be stationary
        self.initial_radius = self.radius  # Track original size for visual effects
        self.eroded = False  # Track if obstacle has been eroded
        self.min_radius = 8  # Minimum size before disappearing
        
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
            
        # Draw obstacle as a rock with texture
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
        
        pygame.draw.circle(surface, rock_color, (int(self.position[0]), int(self.position[1])), self.radius)
        
        # Add darker shading on one side
        pygame.draw.arc(surface, GRAY, 
                      (int(self.position[0] - self.radius), int(self.position[1] - self.radius),
                       self.radius * 2, self.radius * 2),
                      math.pi/4, math.pi, max(2, self.radius//2))
                      
        # Add cracks if eroded
        if self.eroded:
            # Draw some random cracks
            crack_start = self.position
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
            pygame.draw.circle(surface, (200, 100, 50), (int(indicator_x), int(indicator_y)), 3)
        
    def collides_with(self, x, y, radius):
        dx = self.position[0] - x
        dy = self.position[1] - y
        distance = math.sqrt(dx*dx + dy*dy)
        return distance < (self.radius + radius)

class Organism:
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
        self.color_variation = (
            random.randint(-20, 20),
            random.randint(-20, 20),
            random.randint(-20, 20)
        )  # Individual color variation
        
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
            
    def sense_environment(self, foods, obstacles):
        # Return a list of sensory inputs based on vision
        vision_range = self.genome['vision_range']
        field_of_view = self.genome['field_of_view']
        num_sensors = int(self.genome['num_sensors'])
        
        # Initialize sensors (food, obstacle) for each angle
        sensors = [(0, 0) for _ in range(num_sensors)]
        
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
            
            # Check for food along the ray
            closest_food_dist = vision_range
            for food in foods:
                if food.active:
                    food_dist = math.sqrt((food.position[0] - self.position[0])**2 + 
                                        (food.position[1] - self.position[1])**2)
                    
                    if food_dist < closest_food_dist:
                        # Check if food is in the direction of the ray
                        food_angle = math.atan2(food.position[1] - self.position[1], 
                                              food.position[0] - self.position[0])
                        angle_diff = abs(angle - food_angle)
                        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
                        
                        if angle_diff < field_of_view / num_sensors:
                            closest_food_dist = food_dist
                            sensors[i] = (1 - food_dist/vision_range, sensors[i][1])
            
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
                        sensors[i] = (sensors[i][0], 1 - obstacle_dist/vision_range)
        
        return sensors
        
    def make_decision(self, sensors):
        # Simple decision making based on sensor input
        turn_strength = 0
        speed_change = 0
        
        # Process each sensor
        for i, (food_signal, obstacle_signal) in enumerate(sensors):
            sensor_angle = -self.genome['field_of_view']/2 + i * self.genome['field_of_view']/(len(sensors)-1 if len(sensors) > 1 else 1)
            
            # Food attraction
            turn_strength += sensor_angle * food_signal * self.genome['food_attraction']
            speed_change += food_signal * self.genome['food_attraction']
            
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
                    self.energy += food.energy
                    self.food_eaten += 1
                    food.active = False
        
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
                # Regular dead organism as an "X"
                size = int(self.radius * 0.8)
                color = (100, 100, 100, alpha)
                
                pygame.draw.line(death_surf, color, 
                               (x - size, y - size), (x + size, y + size), 2)
                pygame.draw.line(death_surf, color, 
                               (x + size, y - size), (x - size, y + size), 2)
                
            surface.blit(death_surf, (0, 0))
            return
            
        # Draw body as a small creature with eyes
        
        # Calculate color based on base + variation
        base_color = BLUE  # Base blue color
        body_color = (
            max(0, min(255, base_color[0] + self.color_variation[0])),
            max(0, min(255, base_color[1] + self.color_variation[1])),
            max(0, min(255, base_color[2] + self.color_variation[2]))
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
        
        # Draw small generation indicator (only in detailed mode)
        if not FAST_MODE and self.generation > 1:
            gen_text = str(self.generation)
            small_font = pygame.font.SysFont(None, 14)
            gen_surf = small_font.render(gen_text, True, WHITE)
            surface.blit(gen_surf, (x - 3, y - self.radius - 20))
    
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
        new_genome['reproduction_threshold'] = max(600, min(1200, new_genome['reproduction_threshold']))
        new_genome['offspring_energy'] = max(0.1, min(0.5, new_genome['offspring_energy']))
        new_genome['size_factor'] = max(0.6, min(1.5, new_genome['size_factor']))
        
        # Create a new organism with the mutated genome
        # Note: Position and parent will be set properly when reproduction actually happens
        return Organism(new_genome)

def calculate_fitness(organism):
    # The fitness function considers:
    # 1. How much food was eaten
    # 2. How much energy remains
    # 3. How long the organism survived
    
    fitness = organism.food_eaten * 100  # Primary objective
    
    if organism.alive:
        fitness += organism.energy * 0.1  # Bonus for remaining energy
    
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

def simulate_continuous_evolution(screen):
    # Setup environment
    population = [Organism() for _ in range(POPULATION_SIZE)]
    foods = [Food() for _ in range(FOOD_COUNT)]
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
                            # Create food at the organism's position
                            new_food = Food(position=organism.position)
                            new_food.energy = 150  # More nutritious than regular food
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
                    foods.append(Food())
        
        # Track statistics
        if step % stat_collection_interval == 0:
            if population:
                avg_generation = sum(o.generation for o in population) / len(population)
                max_generation = max(o.generation for o in population)
                highest_generation = max(highest_generation, max_generation)
                
                stats = {
                    'step': step,
                    'population': len(population),
                    'alive_count': alive_count,
                    'food_count': active_food_count,
                    'avg_generation': avg_generation,
                    'max_generation': max_generation
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
    
    # Calculate statistics
    alive_count = len(alive_organisms)
    total_count = len(population)
    active_food = sum(1 for food in foods if food.active)
    
    max_gen = max((o.generation for o in population), default=1)
    avg_gen = sum(o.generation for o in population) / max(1, len(population))
    
    # Display statistics
    stats_text = [
        f"Step: {step} | Population: {alive_count}/{total_count} | Food: {active_food}/{len(foods)}",
        f"Generations: Avg {avg_gen:.1f} | Max {max_gen} | Obstacles: {len(obstacles)}",
        f"{'FAST MODE' if FAST_MODE else 'DETAILED MODE'} - Press SPACE to toggle | Press P to pause"
    ]
    
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
        
        # Draw data points and lines for population and generation
        pop_points = []
        gen_points = []
        
        for i, stat in enumerate(stats):
            x = graph_margin + (stat['step'] / max_steps) * graph_width
            
            # Population (blue)
            pop_y = graph_margin + graph_height - (stat['population'] / max_population * graph_height * 0.8)
            pop_points.append((x, pop_y))
            
            # Maximum generation (red)
            gen_y = graph_margin + graph_height - (stat['max_generation'] / max_generation * graph_height * 0.8)
            gen_points.append((x, gen_y))
            
        # Draw lines connecting points
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
                                    
        # Draw legend
        pygame.draw.line(screen, (255, 0, 0), (graph_margin + graph_width + 10, graph_margin + 20),
                       (graph_margin + graph_width + 40, graph_margin + 20), 2)
        pygame.draw.line(screen, (0, 0, 255), (graph_margin + graph_width + 10, graph_margin + 50),
                       (graph_margin + graph_width + 40, graph_margin + 50), 2)
                       
        gen_label = font.render("Max Generation", True, BLACK)
        pop_label = font.render("Population", True, BLACK)
        
        screen.blit(gen_label, (graph_margin + graph_width + 50, graph_margin + 10))
        screen.blit(pop_label, (graph_margin + graph_width + 50, graph_margin + 40))
        
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