# Evolve - Artificial Life Simulation

## Overview

Evolve is a digital ecosystem simulator that models evolutionary processes in a virtual environment. The project includes two main simulations:

1. **Organism Evolution** (`organism_evolution.py`) - A complex ecosystem where organisms move, sense, eat, reproduce, and evolve in real-time
2. **Shape Evolution** (`shape_evolution.py`) - A simpler simulation where shapes evolve to match target patterns

## Organism Evolution Features

The primary simulation models a complete ecosystem with the following elements:

### Environment
- Food resources that grow and become more nutritious over time
- Obstacles that can be eroded through interaction
- Continuous, persistent world (no discrete generations)
- Complete ecosystem cycle (food → organisms → waste → obstacles)

### Organisms
- Sensory systems with variable vision range and field of view
- Neural-like decision making for food seeking and obstacle avoidance
- Metabolism, energy management, and natural death
- Reproduction when energy thresholds are met
- Genetic mutations that affect behavior and physical traits
- Species identification based on genetic similarity

### Visual Interface
- Real-time visualization of the ecosystem
- Color coding of different species
- Statistics tracking for population, generations, and species
- Energy and health indicators for organisms
- Fast/detailed view modes

### Evolution Tracking
- Species emergence and extinction tracking
- Performance graphs for dominant species over time
- Fitness metrics for different genetic strategies

## Shape Evolution Features

A secondary simulation focused on shape pattern evolution:

- Target-based evolution toward predefined shapes
- Multiple shape types (star, circle, spiral, etc.)
- Visual representation of fitness improvement over generations

## Installation

Requirements:
- Python 3.6+
- Pygame
- NumPy

Install dependencies:
```
pip install -r requirements.txt
```

## Running the Simulations

To run the organism evolution simulation:
```
python organism_evolution.py
```

To run the shape evolution simulation:
```
python shape_evolution.py
```

## Controls

- **Space**: Toggle between fast mode and detailed mode
- **P**: Pause/resume simulation

## TODO - Future Enhancements

### Environmental Complexity
- [ ] Add seasonal changes where food availability fluctuates
- [ ] Introduce different food types with varying nutritional value
- [ ] Create varied terrain (water, land, mountains) with different movement costs
- [ ] Implement weather patterns that affect organism movement/energy consumption
- [ ] Add day/night cycles that affect visibility and behavior

### More Complex Organisms
- [ ] Add more detailed genomes with interdependent traits
- [ ] Implement age-related changes (juveniles, adults, elderly)
- [ ] Add separate male/female organisms with sexual reproduction
- [ ] Include predator-prey relationships between species
- [ ] Implement specialized organs that provide specific advantages

### Improved Ecological Interactions
- [ ] Add symbiotic relationships between species
- [ ] Implement more realistic energy transfer through food web
- [ ] Add resource competition (not just for food, but for space/shelter)
- [ ] Introduce diseases or parasites that can spread
- [ ] Model social behaviors (grouping, competition, cooperation)

### Advanced Neural Networks
- [ ] Replace simple decision-making with evolvable neural networks
- [ ] Allow organisms to learn during their lifetime
- [ ] Implement memory of past experiences
- [ ] Add more complex sensory inputs (smell, hearing, etc.)
- [ ] Create communication mechanisms between organisms

### Richer Statistics & Visualization
- [ ] Track genealogy trees for individuals
- [ ] Show evolutionary pressure maps on the environment
- [ ] Offer more detailed analysis of which traits led to success
- [ ] Create exportable data for external analysis
- [ ] Add 3D visualization options

### User Interface Improvements
- [ ] Create interactive tools to modify the environment during simulation
- [ ] Add ability to save/load ecosystem states
- [ ] Implement time controls (speed up, slow down, rewind)
- [ ] Add a "specimen view" to examine individual organisms in detail
- [ ] Create a graphical parameter editor for simulation settings

## Science Background

This simulation is based on principles from:
- Darwinian evolution and natural selection
- Population genetics
- Ecology and ecosystem dynamics
- Artificial life research
- Emergent behavior in complex systems

The goal is to observe how complex behaviors and specialized adaptations can emerge from simple rules and random variation when subjected to selection pressures.