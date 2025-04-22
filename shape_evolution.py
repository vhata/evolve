import turtle
import random
import math
from typing import List, Tuple

# Constants
WINDOW_SIZE = 800
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.1
SHAPE_POINTS = 8  # Number of points in our evolving shape


class Shape:
    def __init__(self, points=None):
        if points is None:
            # Initialize with random points
            self.points = [
                (random.randint(-300, 300), random.randint(-300, 300))
                for _ in range(SHAPE_POINTS)
            ]
        else:
            self.points = points
        self.fitness = 0

    def mutate(self):
        mutated_points = []
        for point in self.points:
            if random.random() < MUTATION_RATE:
                # Add some random variation to the point
                new_x = max(-300, min(300, point[0] + random.randint(-20, 20)))
                new_y = max(-300, min(300, point[1] + random.randint(-20, 20)))
                mutated_points.append((new_x, new_y))
            else:
                mutated_points.append(point)
        return Shape(mutated_points)

    def draw(self, t: turtle.Turtle, color="black"):
        t.penup()
        t.goto(self.points[0])
        t.pendown()
        t.color(color)
        for point in self.points[1:]:
            t.goto(point)
        t.goto(self.points[0])  # Close the shape


def create_target_shape() -> Shape:
    # Create a star-like shape as our target
    points = []
    for i in range(SHAPE_POINTS):
        angle = (2 * math.pi * i) / SHAPE_POINTS
        if i % 2 == 0:
            r = 100
        else:
            r = 50
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        points.append((x, y))
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
    target = create_target_shape()

    generation = 0

    while generation < GENERATIONS:
        # Clear the screen
        t.clear()

        # Draw target shape in red
        target.draw(t, "red")

        # Draw best shape in black
        population[0].draw(t, "black")

        # Display generation number
        t.penup()
        t.goto(-380, 350)
        t.write(f"Generation: {generation}", font=("Arial", 16, "normal"))

        # Update the screen
        screen.update()

        # Evolve the population
        population = evolve_population(population, target)
        generation += 1

        # Small delay to see the evolution
        turtle.delay(50)

    screen.exitonclick()


if __name__ == "__main__":
    main()
