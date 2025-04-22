import os
import time
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import datetime

class StatsTracker:
    """Track and record statistics for the evolution simulation."""
    
    def __init__(self, output_dir="stats", session_id=None):
        """
        Initialize the stats tracker.
        
        Args:
            output_dir: Directory to store statistics files
            session_id: Unique identifier for this simulation run (defaults to timestamp)
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate session ID if not provided
        if session_id is None:
            self.session_id = f"sim_{int(time.time())}"
        else:
            self.session_id = session_id
            
        # Initialize storage for various statistics
        self.time_series = []  # For time-series data
        self.species_data = defaultdict(list)  # For species-specific data
        self.food_data = []  # For food-related statistics
        self.event_log = []  # For important events
        
        # Metadata about this simulation run
        self.metadata = {
            "start_time": time.time(),
            "session_id": self.session_id,
            "parameters": {}  # Will be filled with simulation parameters
        }
        
        # Setup file paths
        self.time_series_path = os.path.join(output_dir, f"{self.session_id}_timeseries.csv")
        self.species_data_path = os.path.join(output_dir, f"{self.session_id}_species.csv")
        self.food_data_path = os.path.join(output_dir, f"{self.session_id}_food.csv")
        self.event_log_path = os.path.join(output_dir, f"{self.session_id}_events.json")
        self.metadata_path = os.path.join(output_dir, f"{self.session_id}_metadata.json")
        
        # Initialize CSV headers
        self._initialize_csv_files()
    
    def _initialize_csv_files(self):
        """Set up CSV files with headers."""
        # Time series data headers
        with open(self.time_series_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'timestamp', 'population', 'alive_count', 'food_count',
                'obstacle_count', 'avg_generation', 'max_generation', 'species_count',
                'avg_energy', 'avg_size', 'avg_speed', 'avg_vision_range', 'avg_metabolism'
            ])
        
        # Species data headers
        with open(self.species_data_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'timestamp', 'species_id', 'count', 'avg_fitness', 
                'avg_energy', 'avg_age', 'std_food_pref', 'rich_food_pref', 
                'super_food_pref', 'junk_food_pref', 'specialized_food',
                'avg_metabolism', 'avg_size', 'avg_speed', 'avg_vision_range'
            ])
            
        # Food data headers
        with open(self.food_data_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'timestamp', 'total_food', 'standard_food', 'rich_food', 
                'superfood', 'junk_food', 'avg_food_energy'
            ])
    
    def record_parameters(self, params_dict):
        """Record the simulation parameters."""
        self.metadata["parameters"] = params_dict
        
        # Write to metadata file
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def record_time_series(self, step, stats_dict):
        """Record time series data for the current step."""
        # Add timestamp and step to stats
        stats_dict['timestamp'] = time.time()
        stats_dict['step'] = step
        
        # Store in memory
        self.time_series.append(stats_dict)
        
        # Write to CSV
        with open(self.time_series_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Ensure we have all fields, using None for missing ones
            row = [
                stats_dict.get('step'),
                stats_dict.get('timestamp'),
                stats_dict.get('population', 0),
                stats_dict.get('alive_count', 0),
                stats_dict.get('food_count', 0),
                stats_dict.get('obstacle_count', 0),
                stats_dict.get('avg_generation', 0),
                stats_dict.get('max_generation', 0),
                stats_dict.get('species_count', 0),
                stats_dict.get('avg_energy', 0),
                stats_dict.get('avg_size', 0),
                stats_dict.get('avg_speed', 0),
                stats_dict.get('avg_vision_range', 0),
                stats_dict.get('avg_metabolism', 0)
            ]
            writer.writerow(row)
    
    def record_species_data(self, step, species_stats_list):
        """Record data for each species."""
        timestamp = time.time()
        
        with open(self.species_data_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            for species_stats in species_stats_list:
                species_id = species_stats.get('species_id')
                
                # Store in memory
                self.species_data[species_id].append(species_stats)
                
                # Create CSV row
                row = [
                    step,
                    timestamp,
                    species_id,
                    species_stats.get('count', 0),
                    species_stats.get('avg_fitness', 0),
                    species_stats.get('avg_energy', 0),
                    species_stats.get('avg_age', 0),
                    species_stats.get('std_food_pref', 0),
                    species_stats.get('rich_food_pref', 0),
                    species_stats.get('super_food_pref', 0),
                    species_stats.get('junk_food_pref', 0),
                    species_stats.get('specialized_food'),
                    species_stats.get('avg_metabolism', 0),
                    species_stats.get('avg_size', 0),
                    species_stats.get('avg_speed', 0),
                    species_stats.get('avg_vision_range', 0)
                ]
                writer.writerow(row)
    
    def record_food_data(self, step, food_stats):
        """Record food distribution data."""
        # Add timestamp and step to stats
        food_stats['timestamp'] = time.time()
        food_stats['step'] = step
        
        # Store in memory
        self.food_data.append(food_stats)
        
        # Write to CSV
        with open(self.food_data_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Create CSV row
            row = [
                step,
                food_stats.get('timestamp'),
                food_stats.get('total_food', 0),
                food_stats.get('standard_food', 0),
                food_stats.get('rich_food', 0),
                food_stats.get('superfood', 0),
                food_stats.get('junk_food', 0),
                food_stats.get('avg_food_energy', 0)
            ]
            writer.writerow(row)
    
    def log_event(self, event_type, event_data=None):
        """Log a significant event in the simulation."""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'data': event_data or {}
        }
        
        # Store in memory
        self.event_log.append(event)
        
        # Write to JSON file (append)
        with open(self.event_log_path, 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    def generate_report(self, output_path=None):
        """Generate an HTML report with visualizations of the statistics."""
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"{self.session_id}_report.html")
            
        # Generate all the needed visualizations
        fig_timeseries = self._create_time_series_plot()
        fig_species = self._create_species_plot()
        fig_food = self._create_food_plot()
        
        # Save the figures
        figs_dir = os.path.join(self.output_dir, f"{self.session_id}_figures")
        os.makedirs(figs_dir, exist_ok=True)
        
        fig_timeseries.savefig(os.path.join(figs_dir, "timeseries.png"))
        fig_species.savefig(os.path.join(figs_dir, "species.png"))
        fig_food.savefig(os.path.join(figs_dir, "food.png"))
        
        # Generate HTML report
        metadata_display = ""
        for key, value in self.metadata.get("parameters", {}).items():
            metadata_display += f"<tr><td>{key}</td><td>{value}</td></tr>"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Simulation Report: {self.session_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
                .stats-card {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .full-width {{ grid-column: 1 / -1; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .chart {{ margin-bottom: 30px; max-width: 100%; }}
                .chart img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Simulation Report</h1>
                <p>Session ID: {self.session_id}</p>
                <p>Start Time: {datetime.datetime.fromtimestamp(self.metadata.get("start_time", 0)).strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Simulation Parameters</h2>
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                    {metadata_display}
                </table>
                
                <h2>Simulation Statistics</h2>
                
                <h3>Population and Food</h3>
                <div class="chart">
                    <img src="{self.session_id}_figures/timeseries.png" alt="Time Series Data">
                </div>
                
                <h3>Species Evolution</h3>
                <div class="chart">
                    <img src="{self.session_id}_figures/species.png" alt="Species Data">
                </div>
                
                <h3>Food Distribution</h3>
                <div class="chart">
                    <img src="{self.session_id}_figures/food.png" alt="Food Distribution">
                </div>
                
                <h2>Key Events</h2>
                <ul>
        """
        
        # Add key events to the report
        recent_events = self.event_log[-20:] if len(self.event_log) > 20 else self.event_log
        for event in recent_events:
            event_time = datetime.datetime.fromtimestamp(event.get("timestamp", 0)).strftime('%Y-%m-%d %H:%M:%S')
            html_content += f"<li>{event_time}: {event.get('type')} - {event.get('data', {})}</li>"
        
        html_content += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        return output_path
    
    def _create_time_series_plot(self):
        """Create a time series plot of key statistics."""
        if not self.time_series:
            # Return empty figure if no data
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No time series data available", ha='center', va='center')
            return fig
            
        # Extract data
        steps = [entry.get('step', 0) for entry in self.time_series]
        population = [entry.get('population', 0) for entry in self.time_series]
        food_count = [entry.get('food_count', 0) for entry in self.time_series]
        species_count = [entry.get('species_count', 0) for entry in self.time_series]
        avg_energy = [entry.get('avg_energy', 0) for entry in self.time_series]
        
        # Create plot
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot population and food
        axs[0].plot(steps, population, label='Population', color='blue')
        axs[0].plot(steps, food_count, label='Food', color='green')
        axs[0].set_ylabel('Count')
        axs[0].set_title('Population and Food Over Time')
        axs[0].legend(loc='upper left')
        axs[0].grid(True, alpha=0.3)
        
        # Add species count on second axis
        ax2 = axs[0].twinx()
        ax2.plot(steps, species_count, label='Species', color='red', linestyle='--')
        ax2.set_ylabel('Species Count')
        ax2.legend(loc='upper right')
        
        # Plot average energy
        axs[1].plot(steps, avg_energy, label='Avg Energy', color='orange')
        axs[1].set_xlabel('Simulation Step')
        axs[1].set_ylabel('Energy')
        axs[1].set_title('Average Organism Energy Over Time')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_species_plot(self):
        """Create a plot of species dynamics."""
        # Return empty figure if no data
        if not self.species_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No species data available", ha='center', va='center')
            return fig
            
        # Find top species by peak population
        species_peaks = {}
        for species_id, data in self.species_data.items():
            counts = [entry.get('count', 0) for entry in data]
            if counts:
                species_peaks[species_id] = max(counts)
        
        # Sort and get top 5 species
        top_species = sorted(species_peaks.items(), key=lambda x: x[1], reverse=True)[:5]
        top_species_ids = [s[0] for s in top_species]
        
        # Create plot
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Create a single list of steps
        all_steps = set()
        for data in self.species_data.values():
            for entry in data:
                if 'step' in entry:
                    all_steps.add(entry['step'])
        steps = sorted(all_steps)
        
        # Plot top species populations
        for species_id in top_species_ids:
            if species_id in self.species_data:
                # Create a step-to-count mapping
                step_counts = {}
                for entry in self.species_data[species_id]:
                    step = entry.get('step')
                    count = entry.get('count', 0)
                    if step is not None:
                        step_counts[step] = count
                
                # Get counts in order of steps
                counts = [step_counts.get(step, 0) for step in steps]
                axs[0].plot(steps, counts, label=f'Species {species_id}')
        
        axs[0].set_ylabel('Population')
        axs[0].set_title('Top Species Populations Over Time')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        
        # Plot food preferences for top species
        for species_id in top_species_ids:
            if species_id in self.species_data:
                # Get food preference data
                food_prefs = []
                for entry in self.species_data[species_id]:
                    step = entry.get('step')
                    if step is not None and step % 10 == 0:  # Sample every 10 steps
                        prefs = {
                            'standard': entry.get('std_food_pref', 0),
                            'rich': entry.get('rich_food_pref', 0),
                            'super': entry.get('super_food_pref', 0),
                            'junk': entry.get('junk_food_pref', 0)
                        }
                        # Normalize to show relative preferences
                        total = sum(prefs.values())
                        if total > 0:
                            for key in prefs:
                                prefs[key] /= total
                        food_prefs.append((step, prefs))
                
                # Plot if we have data
                if food_prefs:
                    sampled_steps, pref_data = zip(*food_prefs)
                    std_prefs = [p['standard'] for p in pref_data]
                    rich_prefs = [p['rich'] for p in pref_data]
                    super_prefs = [p['super'] for p in pref_data]
                    junk_prefs = [p['junk'] for p in pref_data]
                    
                    # Stack the preferences
                    stack_data = np.vstack([std_prefs, rich_prefs, super_prefs, junk_prefs])
                    axs[1].stackplot(sampled_steps, stack_data, labels=['Standard', 'Rich', 'Super', 'Junk'],
                                    colors=['green', 'gold', 'purple', 'gray'], alpha=0.7)
                    
                    # Only show one legend for the first species
                    if species_id == top_species_ids[0]:
                        axs[1].legend(loc='upper right')
                    
        axs[1].set_xlabel('Simulation Step')
        axs[1].set_ylabel('Relative Preference')
        axs[1].set_title('Food Type Preferences for Dominant Species')
        axs[1].set_ylim(0, 1)
        axs[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_food_plot(self):
        """Create a plot of food distribution over time."""
        if not self.food_data:
            # Return empty figure if no data
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No food data available", ha='center', va='center')
            return fig
            
        # Extract data
        steps = [entry.get('step', 0) for entry in self.food_data]
        standard_food = [entry.get('standard_food', 0) for entry in self.food_data]
        rich_food = [entry.get('rich_food', 0) for entry in self.food_data]
        superfood = [entry.get('superfood', 0) for entry in self.food_data]
        junk_food = [entry.get('junk_food', 0) for entry in self.food_data]
        avg_energy = [entry.get('avg_food_energy', 0) for entry in self.food_data]
        
        # Create plot
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot food type distribution as stacked area chart
        axs[0].stackplot(steps, 
                        standard_food, rich_food, superfood, junk_food,
                        labels=['Standard', 'Rich', 'Super', 'Junk'],
                        colors=['green', 'gold', 'purple', 'gray'], alpha=0.7)
        axs[0].set_ylabel('Count')
        axs[0].set_title('Food Type Distribution Over Time')
        axs[0].legend(loc='upper right')
        axs[0].grid(True, alpha=0.3)
        
        # Plot average food energy
        axs[1].plot(steps, avg_energy, label='Avg Food Energy', color='red')
        axs[1].set_xlabel('Simulation Step')
        axs[1].set_ylabel('Energy')
        axs[1].set_title('Average Food Energy Over Time')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig