"""Utility functions for collecting statistics from the simulation."""

def collect_basic_stats(population, foods, obstacles):
    """Collect basic simulation statistics."""
    # Population stats
    alive_organisms = [o for o in population if o.alive]
    
    # Calculate averages for living organisms
    if alive_organisms:
        avg_energy = sum(o.energy for o in alive_organisms) / len(alive_organisms)
        avg_size = sum(o.genome['size_factor'] for o in alive_organisms) / len(alive_organisms)
        avg_speed = sum(o.speed for o in alive_organisms) / len(alive_organisms)
        avg_vision_range = sum(o.genome['vision_range'] for o in alive_organisms) / len(alive_organisms)
        avg_metabolism = sum(o.genome['metabolism'] for o in alive_organisms) / len(alive_organisms)
        avg_generation = sum(o.generation for o in alive_organisms) / len(alive_organisms)
        max_generation = max(o.generation for o in alive_organisms) if alive_organisms else 0
    else:
        avg_energy = 0
        avg_size = 0
        avg_speed = 0
        avg_vision_range = 0
        avg_metabolism = 0
        avg_generation = 0
        max_generation = 0
    
    # Count unique species
    species_ids = set(o.species_id for o in population if o.species_id is not None)
    
    # Food stats
    active_foods = [f for f in foods if f.active]
    
    stats = {
        'population': len(population),
        'alive_count': len(alive_organisms),
        'food_count': len(active_foods),
        'obstacle_count': len(obstacles),
        'avg_energy': avg_energy,
        'avg_size': avg_size,
        'avg_speed': avg_speed,
        'avg_vision_range': avg_vision_range,
        'avg_metabolism': avg_metabolism,
        'avg_generation': avg_generation,
        'max_generation': max_generation,
        'species_count': len(species_ids)
    }
    
    return stats

def collect_species_stats(population, species_registry):
    """Collect statistics for each active species."""
    # Group organisms by species
    species_organisms = {}
    for organism in population:
        if organism.species_id and organism.alive:
            if organism.species_id not in species_organisms:
                species_organisms[organism.species_id] = []
            species_organisms[organism.species_id].append(organism)
    
    species_stats_list = []
    
    # Calculate stats for each species
    for species_id, organisms in species_organisms.items():
        if species_id in species_registry:
            species_info = species_registry[species_id]
            
            # Basic stats
            count = len(organisms)
            avg_fitness = sum(organism.fitness for organism in organisms) / count if count > 0 else 0
            avg_energy = sum(organism.energy for organism in organisms) / count if count > 0 else 0
            avg_age = sum(organism.age for organism in organisms) / count if count > 0 else 0
            
            # Genome stats
            avg_metabolism = sum(o.genome['metabolism'] for o in organisms) / count if count > 0 else 0
            avg_size = sum(o.genome['size_factor'] for o in organisms) / count if count > 0 else 0
            avg_speed = sum(o.speed for o in organisms) / count if count > 0 else 0
            avg_vision_range = sum(o.genome['vision_range'] for o in organisms) / count if count > 0 else 0
            
            # Food preferences
            avg_std_food_pref = sum(o.genome['standard_food_pref'] for o in organisms) / count if count > 0 else 0
            avg_rich_food_pref = sum(o.genome['rich_food_pref'] for o in organisms) / count if count > 0 else 0
            avg_super_food_pref = sum(o.genome['superfood_pref'] for o in organisms) / count if count > 0 else 0
            avg_junk_food_pref = sum(o.genome['junk_food_pref'] for o in organisms) / count if count > 0 else 0
            
            # Get specialized food if available
            specialized_food = species_info.get('specialized_food')
            
            species_stats = {
                'species_id': species_id,
                'count': count,
                'avg_fitness': avg_fitness,
                'avg_energy': avg_energy,
                'avg_age': avg_age,
                'avg_metabolism': avg_metabolism,
                'avg_size': avg_size,
                'avg_speed': avg_speed,
                'avg_vision_range': avg_vision_range,
                'std_food_pref': avg_std_food_pref,
                'rich_food_pref': avg_rich_food_pref,
                'super_food_pref': avg_super_food_pref,
                'junk_food_pref': avg_junk_food_pref,
                'specialized_food': specialized_food
            }
            
            species_stats_list.append(species_stats)
    
    return species_stats_list

def collect_food_stats(foods):
    """Collect statistics about food distribution."""
    active_foods = [f for f in foods if f.active]
    
    # Initialize counters for food types
    food_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Standard, Rich, Superfood, Junk
    
    # Count food types and calculate average energy
    total_energy = 0
    for food in active_foods:
        food_counts[food.food_type] += 1
        total_energy += food.energy
    
    avg_food_energy = total_energy / len(active_foods) if active_foods else 0
    
    stats = {
        'total_food': len(active_foods),
        'standard_food': food_counts[0],
        'rich_food': food_counts[1],
        'superfood': food_counts[2],
        'junk_food': food_counts[3],
        'avg_food_energy': avg_food_energy
    }
    
    return stats