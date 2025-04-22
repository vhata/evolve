import turtle
import random
import math
import time
from typing import List, Tuple, Callable, Dict
from enum import Enum

# Constants
WINDOW_SIZE = 800
POPULATION_SIZE = 50
GENERATIONS = 5000
MUTATION_RATE = 0.1  # Probability of small mutation per point
LARGE_MUTATION_PROBABILITY = 0.05 # Probability of large mutation per point
LARGE_MUTATION_MAGNITUDE = 80 # Max change for large mutation
SHAPE_POINTS = 80  # Number of points in our evolving shape
TARGET_RADIUS = 150 # General radius for target shapes

# Define Shape Types (Using integers for potential future mapping)
class ShapeType(Enum):
    STAR = 1
    CIRCLE = 2
    POLYGON = 3
    STAR_BURST = 4
    SPIRAL = 5
    POLYGON_MORPH = 6
    WAVE_PATTERN = 7
    FRACTAL = 8 # Simple interpretation
    BUTTERFLY = 9
    ABSTRACT = 10



class Shape:
    def __init__(self, points=None):
        if points is None:
            # Initialize with random points
            self.points = [
                (random.randint(-350, 350), random.randint(-350, 350))
                for _ in range(SHAPE_POINTS)
            ]
        else:
            self.points = points
        self.fitness = 0

    def mutate(self):
        mutated_points = []
        for x, y in self.points: # Unpack points directly
            change_x, change_y = 0, 0 # Default to no change

            # Determine mutation type and magnitude
            rand_val = random.random()
            if rand_val < LARGE_MUTATION_PROBABILITY:
                # Large mutation
                change_x = random.randint(-LARGE_MUTATION_MAGNITUDE, LARGE_MUTATION_MAGNITUDE)
                change_y = random.randint(-LARGE_MUTATION_MAGNITUDE, LARGE_MUTATION_MAGNITUDE)
            elif rand_val < LARGE_MUTATION_PROBABILITY + MUTATION_RATE: # Check combined probability
                # Small mutation (occurs if not large, but within MUTATION_RATE range)
                change_x = random.randint(-20, 20)
                change_y = random.randint(-20, 20)
            # else: No mutation, change_x/y remain 0

            # Apply the calculated change
            new_x = x + change_x
            new_y = y + change_y

            # Clamp the final point to window bounds once
            bound = WINDOW_SIZE // 2 - 50 # Boundary margin
            new_x = max(-bound, min(bound, new_x))
            new_y = max(-bound, min(bound, new_y))
            
            mutated_points.append((new_x, new_y))
            
        return Shape(mutated_points)

    def draw(self, t: turtle.Turtle, color="black"):
        t.penup()
        if not self.points:
            return # Don't draw if no points
        t.goto(self.points[0])
        t.pendown()
        t.color(color)
        # Draw segments between points
        for point in self.points[1:]:
            t.goto(point)
        t.goto(self.points[0]) # Close the shape

# --- Helper functions for creating target shape points --- 

def _create_star_points() -> List[Tuple[float, float]]:
    points = []
    outer_radius = TARGET_RADIUS
    inner_radius = TARGET_RADIUS * 0.5
    num_vertices = SHAPE_POINTS
    if num_vertices < 4: num_vertices = 4
    for i in range(num_vertices):
        angle = (2 * math.pi * i) / num_vertices
        r = outer_radius if i % 2 == 0 else inner_radius
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        points.append((x, y))
    return points

def _create_circle_points() -> List[Tuple[float, float]]:
    points = []
    radius = TARGET_RADIUS
    num_vertices = SHAPE_POINTS
    for i in range(num_vertices):
        angle = (2 * math.pi * i) / num_vertices
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        points.append((x, y))
    return points

def _create_polygon_points() -> List[Tuple[float, float]]:
    points = []
    radius = TARGET_RADIUS
    num_sides = 6
    num_vertices = SHAPE_POINTS
    for i in range(num_vertices):
        segment_angle = 2 * math.pi / num_sides
        base_angle = (2 * math.pi * i) / num_vertices
        target_angle = round(base_angle / segment_angle) * segment_angle
        x = radius * math.cos(target_angle)
        y = radius * math.sin(target_angle)
        next_target_angle = target_angle + segment_angle
        x_next = radius * math.cos(next_target_angle)
        y_next = radius * math.sin(next_target_angle)
        segment_progress = (base_angle % segment_angle) / segment_angle
        x_interp = x + (x_next - x) * segment_progress
        y_interp = y + (y_next - y) * segment_progress
        points.append((x_interp, y_interp))
    return points

def _create_star_burst_points() -> List[Tuple[float, float]]:
    """Star with more points and randomized radius."""
    points = []
    num_vertices = SHAPE_POINTS
    for i in range(num_vertices):
        angle = (2 * math.pi * i) / num_vertices
        # Random radius between 50% and 100% of TARGET_RADIUS
        r = TARGET_RADIUS * random.uniform(0.5, 1.0)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        points.append((x, y))
    return points

def _create_spiral_points() -> List[Tuple[float, float]]:
    points = []
    num_vertices = SHAPE_POINTS
    max_radius = TARGET_RADIUS
    coils = 3
    for i in range(num_vertices):
        angle = (2 * math.pi * coils * i) / num_vertices
        radius = max_radius * (i / (num_vertices - 1)) # Avoid div by zero if num_vertices=1
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        points.append((x, y))
    return points

def _create_polygon_morph_points() -> List[Tuple[float, float]]:
    """Morph between a hexagon and a circle."""
    points = []
    num_sides = 6
    num_vertices = SHAPE_POINTS
    for i in range(num_vertices):
        angle = (2 * math.pi * i) / num_vertices
        # Calculate hexagon vertex angle
        segment_angle = 2 * math.pi / num_sides
        target_angle = round(angle / segment_angle) * segment_angle
        # Interpolate radius between polygon radius and circle radius
        morph_factor = abs(math.sin(angle * num_sides / 2.0)) # Varies 0 to 1 between vertices
        poly_r = TARGET_RADIUS / math.cos(segment_angle / 2.0) # Radius to vertex
        r = poly_r + (TARGET_RADIUS - poly_r) * morph_factor 
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        points.append((x, y))
    return points

def _create_wave_pattern_points() -> List[Tuple[float, float]]:
    """Circle shape with radius modulated by sine waves."""
    points = []
    num_vertices = SHAPE_POINTS
    freq1 = 5 # Frequency of the wave
    amp1 = 0.2 * TARGET_RADIUS # Amplitude of the wave
    freq2 = 13 
    amp2 = 0.1 * TARGET_RADIUS
    for i in range(num_vertices):
        angle = (2 * math.pi * i) / num_vertices
        r = TARGET_RADIUS + amp1 * math.sin(angle * freq1) + amp2 * math.cos(angle * freq2)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        points.append((x, y))
    return points

def _create_fractal_points() -> List[Tuple[float, float]]:
    """Generates points resembling a fractal using modulated radius (simpler approach)."""
    points = []
    num_vertices = SHAPE_POINTS
    freq1, freq2, freq3 = 3, 7, 17 # Different frequencies
    amp1, amp2, amp3 = 0.3, 0.15, 0.08 # Decreasing amplitudes
    for i in range(num_vertices):
        angle = (2 * math.pi * i) / num_vertices
        r = TARGET_RADIUS * (1 + 
                              amp1 * math.cos(angle * freq1) + 
                              amp2 * math.cos(angle * freq2) + 
                              amp3 * math.cos(angle * freq3))
        # Ensure radius is positive
        r = max(r, TARGET_RADIUS * 0.1)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        points.append((x, y))
    return points

def _create_butterfly_points() -> List[Tuple[float, float]]:
    """Points for a butterfly curve (scaled)."""
    points = []
    num_vertices = SHAPE_POINTS
    scale = TARGET_RADIUS / 4 # Scale the curve to fit
    for i in range(num_vertices):
        t = (12 * math.pi * i) / (num_vertices -1) # Parametric range
        # Butterfly curve equations
        x = math.sin(t) * (math.exp(math.cos(t)) - 2 * math.cos(4*t) - math.pow(math.sin(t/12), 5))
        y = math.cos(t) * (math.exp(math.cos(t)) - 2 * math.cos(4*t) - math.pow(math.sin(t/12), 5))
        points.append((x * scale, y * scale))
    return points

def _create_abstract_points() -> List[Tuple[float, float]]:
    """Points generated by a random walk from the center."""
    points = []
    x, y = 0.0, 0.0
    max_step = TARGET_RADIUS * 0.15
    num_vertices = SHAPE_POINTS
    for _ in range(num_vertices):
        angle = random.uniform(0, 2 * math.pi)
        step_size = random.uniform(0, max_step)
        x += step_size * math.cos(angle)
        y += step_size * math.sin(angle)
        # Clamp to reasonable bounds
        x = max(-TARGET_RADIUS*1.5, min(TARGET_RADIUS*1.5, x))
        y = max(-TARGET_RADIUS*1.5, min(TARGET_RADIUS*1.5, y))
        points.append((x, y))
    return points

# --- Mapping from ShapeType to creation function --- 

TargetShapeCreator = Callable[[], List[Tuple[float, float]]]

target_shape_creators: Dict[ShapeType, TargetShapeCreator] = {
    ShapeType.STAR: _create_star_points,
    ShapeType.CIRCLE: _create_circle_points, # Added circle back
    ShapeType.POLYGON: _create_polygon_points,
    ShapeType.STAR_BURST: _create_star_burst_points,
    ShapeType.SPIRAL: _create_spiral_points,
    ShapeType.POLYGON_MORPH: _create_polygon_morph_points,
    ShapeType.WAVE_PATTERN: _create_wave_pattern_points,
    ShapeType.FRACTAL: _create_fractal_points,
    ShapeType.BUTTERFLY: _create_butterfly_points,
    ShapeType.ABSTRACT: _create_abstract_points,
}

def create_target_shape(shape_type: ShapeType) -> Shape:
    """Creates the target shape based on the specified type using a lookup."""
    creator_func = target_shape_creators.get(shape_type)
    
    if creator_func:
        points = creator_func()
        # Ensure the correct number of points if creator returns different amount
        if len(points) != SHAPE_POINTS:
             print(f"Warning: Shape creator for {shape_type} returned {len(points)} points, expected {SHAPE_POINTS}. Adjusting...")
             # Simple resampling: linearly interpolate/subsample points
             if len(points) < 2:
                 points = [(0,0)] * SHAPE_POINTS # Fallback for too few points
             else:
                 old_indices = [j * (len(points)-1) / (SHAPE_POINTS-1) for j in range(SHAPE_POINTS)]
                 new_points = []
                 for idx_f in old_indices:
                     idx0 = int(idx_f)
                     idx1 = min(idx0 + 1, len(points) - 1)
                     interp = idx_f - idx0
                     p0 = points[idx0]
                     p1 = points[idx1]
                     x = p0[0] + (p1[0] - p0[0]) * interp
                     y = p0[1] + (p1[1] - p0[1]) * interp
                     new_points.append((x,y))
                 points = new_points
        return Shape(points)
    else:
        print(f"Warning: Shape type {shape_type} not implemented. Defaulting to STAR.")
        points = _create_star_points() # Default to star
        # Ensure default also has correct points
        if len(points) != SHAPE_POINTS:
             # Apply resampling logic here too if necessary, though star should be correct
             pass 
        return Shape(points)

def calculate_fitness(shape: Shape, target: Shape) -> float:
    # Calculate the fitness based on how close the points are to the target
    total_distance: float = 0
    for p1, p2 in zip(shape.points, target.points):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        total_distance += math.sqrt(dx * dx + dy * dy)
    return 1 / (1 + total_distance)  # Higher fitness is better

def evolve_population(population: List[Shape], target: Shape) -> List[Shape]:
    # Calculate fitness for all shapes
    for shape in population:
        shape.fitness = calculate_fitness(shape, target)

    # Sort by fitness
    population.sort(key=lambda x: x.fitness, reverse=True)

    # Keep the best 20% of the population
    new_population = population[: len(population) // 5]

    # Create new shapes through mutation
    while len(new_population) < POPULATION_SIZE:
        parent = random.choice(new_population)
        new_population.append(parent.mutate())

    return new_population

def main():
    # Set up the turtle screen
    screen = turtle.Screen()
    screen.setup(WINDOW_SIZE, WINDOW_SIZE)
    screen.title("Shape Evolution")
    screen.tracer(0)  # Turn off animation for faster drawing

    # Create turtle for drawing
    t = turtle.Turtle()
    t.speed(0)
    t.hideturtle()

    # Create initial population and target
    population = [Shape() for _ in range(POPULATION_SIZE)]
    # Choose random target type
    target_type = random.choice(list(ShapeType))
    target = create_target_shape(target_type) 

    generation = 0
    history_shapes = []  # Store the best shape from each generation
    MAX_HISTORY = 20     # Max number of historical shapes to show

    t.pensize(4)
    while generation < GENERATIONS:
        # Add current best shape to history before evolving
        if population: # Ensure population is not empty
             history_shapes.append(population[0])
             if len(history_shapes) > MAX_HISTORY:
                 history_shapes.pop(0) # Keep history size limited

        # Clear the screen
        t.clear()

        # Draw historical shapes first (light gray, thin)
        t.color("lightgray")
        for historical_shape in history_shapes:
             historical_shape.draw(t, "lightgray")

        # Draw target shape (red, thick)
        target.draw(t, "red")

        # Draw current best shape (evolving color)
        if population: # Check if population still exists
            progress = generation / GENERATIONS
            # Ensure color component stays within 0-255
            green_component = min(255, int(255 * progress)) 
            # Format as #00GG00, ensuring two hex digits for green
            color = f"#00{green_component:02x}00" 
            population[0].draw(t, color)

        # Display generation number
        t.penup()
        t.goto(-380, 350)
        t.write(f"Generation: {generation}", font=("Arial", 16, "normal"))

        # Display target shape type
        t.goto(-380, 330) # Position below generation number
        t.write(f"Target: {target_type.name}", font=("Arial", 16, "normal"))

        # Update the screen
        screen.update()

        # Evolve the population for the *next* generation
        population = evolve_population(population, target)
        generation += 1

        # Small delay to see the evolution
        time.sleep(0.01)

    screen.exitonclick()

if __name__ == "__main__":
    main()
