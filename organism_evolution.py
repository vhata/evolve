import pygame
import random
import math
import numpy as np
from typing import List, Tuple, Dict

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
POPULATION_SIZE = 50
GENERATIONS = 500
FPS = 60
SIMULATION_STEPS = 1000  # Steps per generation
MUTATION_RATE = 0.1
LARGE_MUTATION_PROBABILITY = 0.05
FOOD_COUNT = 20
OBSTACLE_COUNT = 10

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
    def __init__(self):
        self.position = (
            random.randint(50, WINDOW_WIDTH - 50),
            random.randint(50, WINDOW_HEIGHT - 50)
        )
        self.radius = 5
        self.energy = 100
        self.active = True
        
    def draw(self, surface):
        if self.active:
            # Draw food as a bright green apple-like shape
            pygame.draw.circle(surface, GREEN, self.position, self.radius)
            # Add a small stem
            pygame.draw.line(surface, BROWN, 
                          (self.position[0], self.position[1] - self.radius),
                          (self.position[0] + 3, self.position[1] - self.radius - 3), 2)

class Obstacle:
    def __init__(self):
        self.position = (
            random.randint(50, WINDOW_WIDTH - 50),
            random.randint(50, WINDOW_HEIGHT - 50)
        )
        self.radius = random.randint(20, 40)
        
    def draw(self, surface):
        # Draw obstacle as a rock with texture
        pygame.draw.circle(surface, BROWN, self.position, self.radius)
        # Add darker shading on one side
        pygame.draw.arc(surface, GRAY, 
                      (self.position[0] - self.radius, self.position[1] - self.radius,
                       self.radius * 2, self.radius * 2),
                      math.pi/4, math.pi, self.radius//2)
        
    def collides_with(self, x, y, radius):
        dx = self.position[0] - x
        dy = self.position[1] - y
        distance = math.sqrt(dx*dx + dy*dy)
        return distance < (self.radius + radius)

class Organism:
    def __init__(self, genome=None):
        # Position and physics
        self.position = (
            random.randint(50, WINDOW_WIDTH - 50),
            random.randint(50, WINDOW_HEIGHT - 50)
        )
        self.radius = 10
        self.direction = random.uniform(0, 2 * math.pi)
        self.speed = 0
        self.max_speed = 5
        self.energy = 500
        self.food_eaten = 0
        self.alive = True
        
        # Genome defines the organism's behavior
        if genome is None:
            self.genome = {
                # Sensors
                'vision_range': random.uniform(50, 200),
                'field_of_view': random.uniform(math.pi/4, math.pi),
                'num_sensors': random.randint(3, 7),
                
                # Behavior
                'turn_factor': random.uniform(0.1, 0.5),
                'speed_factor': random.uniform(0.5, 2.0),
                'metabolism': random.uniform(0.1, 0.3),
                
                # Neural weights (simplified)
                'food_attraction': random.uniform(0.5, 2.0),
                'obstacle_avoidance': random.uniform(0.5, 2.0),
                'exploration_drive': random.uniform(0.1, 1.0)
            }
        else:
            self.genome = genome
            
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
        
        return turn_strength, speed_change
        
    def update(self, foods, obstacles):
        if not self.alive:
            return
            
        # Sense the environment
        sensors = self.sense_environment(foods, obstacles)
        
        # Make a decision based on sensory input
        turn, acceleration = self.make_decision(sensors)
        
        # Apply the decision
        self.direction += turn * self.genome['turn_factor']
        self.speed += acceleration * self.genome['speed_factor'] * 0.1
        self.speed = max(0, min(self.max_speed, self.speed))
        
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
        for obstacle in obstacles:
            if obstacle.collides_with(new_x, new_y, self.radius):
                can_move = False
                self.direction += math.pi + random.uniform(-0.5, 0.5)  # Bounce with some randomness
                self.energy -= 10  # Energy penalty for hitting obstacle
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
        
        # Check if dead
        if self.energy <= 0:
            self.alive = False
            
    def draw(self, surface):
        x, y = int(self.position[0]), int(self.position[1])
        
        if not self.alive:
            # Draw dead organism as a gray "X"
            size = int(self.radius * 0.8)
            pygame.draw.line(surface, (100, 100, 100), 
                           (x - size, y - size), (x + size, y + size), 2)
            pygame.draw.line(surface, (100, 100, 100), 
                           (x + size, y - size), (x - size, y + size), 2)
            return
            
        # Draw body as a small creature with eyes
        
        # Main body (blue oval)
        pygame.draw.ellipse(surface, BLUE, 
                          (x - self.radius, y - self.radius*0.8, 
                           self.radius*2, self.radius*1.6))
        
        # Direction as a small head/protrusion
        head_x = x + math.cos(self.direction) * self.radius * 0.8
        head_y = y + math.sin(self.direction) * self.radius * 0.8
        pygame.draw.circle(surface, BLUE, (int(head_x), int(head_y)), int(self.radius * 0.5))
        
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
        
        # Draw vision field - makes it easier to understand organism behavior
        if True:  # Set to False to hide vision field
            vision_range = self.genome['vision_range']
            field_of_view = self.genome['field_of_view']
            start_angle = self.direction - field_of_view / 2
            end_angle = self.direction + field_of_view / 2
            
            # Create a transparent surface for the vision cone
            vision_surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            
            # Convert floating point values to integers for the rectangle
            rect_x = int(self.position[0] - vision_range)
            rect_y = int(self.position[1] - vision_range)
            rect_width = int(vision_range * 2)
            rect_height = int(vision_range * 2)
            
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
            pygame.draw.polygon(vision_surf, (255, 255, 0, 30), vision_points)
            
            # Draw the edges of the vision cone
            pygame.draw.line(vision_surf, (255, 255, 0, 60), 
                           (int(self.position[0]), int(self.position[1])), 
                           (int(edge1_x), int(edge1_y)), 2)
            pygame.draw.line(vision_surf, (255, 255, 0, 60), 
                           (int(self.position[0]), int(self.position[1])), 
                           (int(edge2_x), int(edge2_y)), 2)
            
            surface.blit(vision_surf, (0, 0))
        
        # Draw energy bar
        bar_width = 20
        bar_height = 5
        energy_percentage = self.energy / 1000  # Assuming 1000 is max energy
        
        # Background for energy bar (gray)
        pygame.draw.rect(surface, GRAY, 
                       (self.position[0] - bar_width/2, 
                        self.position[1] - self.radius - 10,
                        bar_width,
                        bar_height))
                        
        # Actual energy level (red)
        pygame.draw.rect(surface, RED, 
                       (self.position[0] - bar_width/2, 
                        self.position[1] - self.radius - 10,
                        bar_width * min(1, max(0, energy_percentage)), 
                        bar_height))
    
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
        
        # Create a new organism with the mutated genome
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
        
        # Update organisms
        for organism in population:
            organism.update(foods, obstacles)
        
        # Count active organisms and foods
        alive_count = sum(1 for organism in population if organism.alive)
        food_count = sum(1 for food in foods if food.active)
        
        # End early if all organisms die or all food is eaten
        if alive_count == 0 or food_count == 0:
            break
            
        # Draw everything
        screen.fill(WHITE)
        
        # Draw obstacles and food
        for obstacle in obstacles:
            obstacle.draw(screen)
        for food in foods:
            food.draw(screen)
            
        # Draw organisms
        for organism in population:
            organism.draw(screen)
            
        # Display statistics
        current_avg = sum(o.food_eaten for o in population) / len(population)
        current_best = max(o.food_eaten for o in population)
        best_score = max(best_score, current_best)
        avg_score = current_avg
        
        status_text = f"Step: {step}/{SIMULATION_STEPS} | Alive: {alive_count}/{POPULATION_SIZE} | Food: {food_count}/{FOOD_COUNT}"
        score_text = f"Avg Food: {avg_score:.2f} | Best Food: {best_score}"
        
        status_surface = font.render(status_text, True, BLACK)
        score_surface = font.render(score_text, True, BLACK)
        
        screen.blit(status_surface, (10, 10))
        screen.blit(score_surface, (10, 40))
        
        # Update display
        pygame.display.flip()
        clock.tick(FPS)
    
    # End of generation - calculate and return statistics
    return {
        'best_score': best_score,
        'avg_score': avg_score,
        'alive_count': alive_count
    }

def main():
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Organism Evolution")
    
    # Initialize font
    font = pygame.font.SysFont(None, 24)
    
    # Create initial population
    population = [Organism() for _ in range(POPULATION_SIZE)]
    
    # Track stats across generations
    generation_stats = []
    
    # Main evolution loop
    running = True
    generation = 0
    
    while running and generation < GENERATIONS:
        # Display generation info
        screen.fill(WHITE)
        gen_text = f"Generation {generation + 1} / {GENERATIONS}"
        text_surface = font.render(gen_text, True, BLACK)
        screen.blit(text_surface, (WINDOW_WIDTH // 2 - text_surface.get_width() // 2, WINDOW_HEIGHT // 2))
        pygame.display.flip()
        pygame.time.wait(1000)  # Pause between generations
        
        # Run simulation for current generation
        stats = simulate_generation(population, screen)
        generation_stats.append(stats)
        
        # Display end of generation stats
        screen.fill(WHITE)
        stats_text = [
            f"Generation {generation + 1} Complete",
            f"Best Score: {stats['best_score']}",
            f"Average Score: {stats['avg_score']:.2f}",
            f"Organisms Alive: {stats['alive_count']}/{POPULATION_SIZE}"
        ]
        
        for i, text in enumerate(stats_text):
            text_surface = font.render(text, True, BLACK)
            screen.blit(text_surface, (WINDOW_WIDTH // 2 - text_surface.get_width() // 2, 
                                     WINDOW_HEIGHT // 2 - 50 + i * 30))
        
        pygame.display.flip()
        pygame.time.wait(2000)  # Show stats for 2 seconds
        
        # Process events during pause
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
                
        # Evolve population for next generation
        if running:
            population = evolve_population(population)
            generation += 1
    
    # Final stats display
    if generation_stats:
        screen.fill(WHITE)
        
        # Create a line graph of performance over generations
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
                      
        # Find max value for scaling
        max_score = max(max(s['best_score'] for s in generation_stats), 0.001)
        
        # Draw data points and lines
        best_points = []
        avg_points = []
        
        for i, stats in enumerate(generation_stats):
            x = graph_margin + i * graph_width / len(generation_stats)
            
            best_y = graph_margin + graph_height - (stats['best_score'] / max_score * graph_height)
            avg_y = graph_margin + graph_height - (stats['avg_score'] / max_score * graph_height)
            
            best_points.append((x, best_y))
            avg_points.append((x, avg_y))
            
        # Draw lines connecting points
        if len(best_points) > 1:
            pygame.draw.lines(screen, (255, 0, 0), False, best_points, 2)
            pygame.draw.lines(screen, (0, 0, 255), False, avg_points, 2)
            
        # Draw axis labels
        title = font.render("Evolution Progress", True, BLACK)
        x_label = font.render("Generation", True, BLACK)
        y_label = font.render("Score", True, BLACK)
        
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
                       
        best_label = font.render("Best Score", True, BLACK)
        avg_label = font.render("Average Score", True, BLACK)
        
        screen.blit(best_label, (graph_margin + graph_width + 50, graph_margin + 10))
        screen.blit(avg_label, (graph_margin + graph_width + 50, graph_margin + 40))
        
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