#!/usr/bin/env python3
"""
AIOS IO Simulation - AE=C=1 and Law of Three Fractal Spawning

This simulation implements a mathematical model based on the core claims of the 
AIOS-IO-Global repository:

1. AE = C = 1: Activity Energy equals Consumption equals 1 (normalized baseline)
2. Law of Three: Agents can spawn up to 3 children when resources are abundant
3. RBY Color System: Red/Blue/Yellow consciousness states that cycle
4. Compression Sweeps: Periodic cleanup when excretions exceed threshold
5. Stochastic Mutations: Small parameter variations to simulate evolution

Mathematical Model:
- Each agent has AE (activity energy), C (consumption rate), bloom_score
- Global resource pool R that agents consume from and contribute excretions to
- Spawning probability based on resource abundance and bloom score
- RBY state determines agent behavior and visualization color
- Compression events reduce stored excretions and affect bloom scores

Equations:
- Energy Balance: AE = C = 1 (baseline, can vary with mutations)
- Resource Dynamics: R(t+1) = R(t) - sum(consumption) + sum(excretions)
- Spawning Condition: if R > threshold and bloom > min_bloom and children < 3
- Bloom Evolution: bloom *= (1 + excretion_success - energy_cost)
- RBY Cycling: (R,B,Y) rotates based on timestep and agent state
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd


@dataclass
class Agent:
    """
    Individual agent in the AIO simulation.
    
    Core Properties:
    - ae: Activity Energy (baseline = 1.0)
    - c: Consumption rate (baseline = 1.0) 
    - bloom_score: Fitness/success metric
    - rby_state: Current Red/Blue/Yellow phase (0=R, 1=B, 2=Y)
    - age: Timesteps since creation
    - children_count: Number of offspring spawned (max 3 per Law of Three)
    """
    agent_id: int
    ae: float = 1.0  # Activity Energy
    c: float = 1.0   # Consumption rate
    bloom_score: float = 1.0
    rby_state: int = 0  # 0=Red, 1=Blue, 2=Yellow
    age: int = 0
    children_count: int = 0
    position: Tuple[float, float] = field(default_factory=lambda: (random.random(), random.random()))
    excretion_buffer: float = 0.0
    
    def step(self, global_resources: float) -> Tuple[float, float]:
        """
        Execute one timestep for this agent.
        
        Returns:
        - resource_consumed: Amount consumed from global pool
        - excretion_produced: Amount added to global excretion store
        """
        self.age += 1
        
        # Core equation: AE = C = 1 (with small variations)
        energy_available = self.ae
        consumption_need = self.c
        
        # Calculate actual consumption based on available resources
        consumption_ratio = min(1.0, global_resources / max(consumption_need, 0.1))
        actual_consumption = consumption_need * consumption_ratio
        
        # Activity produces excretions (waste products that can be valuable)
        base_excretion = self.ae * 0.1  # 10% of activity becomes excretion
        bloom_bonus = self.bloom_score * 0.05  # Higher bloom = more valuable excretion
        excretion_produced = base_excretion + bloom_bonus
        
        # Update bloom score based on success
        if consumption_ratio > 0.8:  # Well-fed agents improve
            self.bloom_score *= 1.01
        else:  # Starved agents decline
            self.bloom_score *= 0.99
            
        # Cycle RBY state (Red->Blue->Yellow->Red...)
        self.rby_state = (self.rby_state + 1) % 3
        
        # Store excretion for potential compression
        self.excretion_buffer += excretion_produced
        
        return actual_consumption, excretion_produced
    
    def can_spawn(self, resource_abundance: float) -> bool:
        """Check if agent can spawn offspring according to Law of Three."""
        return (
            self.children_count < 3 and  # Law of Three limit
            self.bloom_score > 0.8 and   # Must be successful
            resource_abundance > 0.6 and # Sufficient resources
            self.age > 5                  # Must be mature
        )
    
    def spawn_child(self, new_id: int) -> 'Agent':
        """Create offspring with inherited traits plus mutation."""
        if not self.can_spawn(1.0):  # Basic check
            raise ValueError("Agent cannot spawn at this time")
            
        self.children_count += 1
        
        # Inherit parent traits with small mutations
        child_ae = self.ae + random.gauss(0, 0.05)  # Small gaussian mutation
        child_c = self.c + random.gauss(0, 0.05)
        child_bloom = self.bloom_score * 0.8  # Children start with reduced bloom
        
        # Keep AE and C close to 1.0 (AE = C = 1 constraint)
        child_ae = max(0.5, min(1.5, child_ae))
        child_c = max(0.5, min(1.5, child_c))
        
        # Random position near parent
        new_x = self.position[0] + random.gauss(0, 0.1)
        new_y = self.position[1] + random.gauss(0, 0.1)
        new_x = max(0, min(1, new_x))  # Keep in bounds
        new_y = max(0, min(1, new_y))
        
        return Agent(
            agent_id=new_id,
            ae=child_ae,
            c=child_c,
            bloom_score=child_bloom,
            rby_state=random.randint(0, 2),  # Random initial RBY state
            position=(new_x, new_y)
        )

    def get_color(self) -> str:
        """Get visualization color based on RBY state."""
        colors = ['red', 'blue', 'yellow']
        return colors[self.rby_state]


def ml_suggest_mutation(agent: Agent) -> Dict[str, float]:
    """
    Stub for ML-driven mutation suggestions.
    
    In a full implementation, this would use machine learning to suggest
    beneficial mutations based on agent performance history.
    
    For now, returns small random variations.
    """
    return {
        'ae_delta': random.gauss(0, 0.02),
        'c_delta': random.gauss(0, 0.02),
        'bloom_delta': random.gauss(0, 0.01)
    }


@dataclass
class SimulationState:
    """Global state of the AIO simulation."""
    agents: List[Agent] = field(default_factory=list)
    global_resources: float = 1000.0  # Global resource pool R
    total_excretions: float = 0.0     # Accumulated excretions
    compression_threshold: float = 500.0  # Trigger compression when exceeded
    timestep: int = 0
    spawn_events: int = 0
    compression_events: int = 0
    next_agent_id: int = 0
    
    def add_agent(self, agent: Optional[Agent] = None) -> Agent:
        """Add new agent to simulation."""
        if agent is None:
            agent = Agent(agent_id=self.next_agent_id)
        self.agents.append(agent)
        self.next_agent_id += 1
        return agent
    
    def step(self) -> Dict[str, Any]:
        """Execute one simulation timestep."""
        self.timestep += 1
        
        # Agents act
        total_consumption = 0.0
        total_excretion = 0.0
        
        for agent in self.agents:
            consumption, excretion = agent.step(self.global_resources)
            total_consumption += consumption
            total_excretion += excretion
        
        # Update global resources
        self.global_resources -= total_consumption
        self.total_excretions += total_excretion
        
        # Resource regeneration (simple model)
        self.global_resources += 10.0  # Base regeneration
        
        # Spawning phase (Law of Three)
        resource_abundance = self.global_resources / max(len(self.agents), 1)
        new_agents = []
        
        for agent in self.agents:
            if agent.can_spawn(resource_abundance / 100.0):
                spawn_prob = 0.1 * (agent.bloom_score / 1.0)  # Higher bloom = higher spawn chance
                if random.random() < spawn_prob:
                    try:
                        child = agent.spawn_child(self.next_agent_id)
                        new_agents.append(child)
                        self.spawn_events += 1
                        # Spawning costs resources
                        self.global_resources -= 50.0
                    except ValueError:
                        pass  # Spawn failed
        
        # Add new agents
        for agent in new_agents:
            self.add_agent(agent)
        
        # Compression sweep when excretions exceed threshold
        if self.total_excretions > self.compression_threshold:
            self.perform_compression()
        
        # Apply stochastic mutations
        self.apply_mutations()
        
        # Collect metrics
        return self.collect_metrics()
    
    def perform_compression(self):
        """
        Compression sweep reduces stored excretions and affects bloom scores.
        
        This represents the system's ability to process and utilize waste products,
        turning them into resources or removing them from the system.
        """
        self.compression_events += 1
        
        # Reduce total excretions
        compression_ratio = 0.7  # Remove 70% of excretions
        compressed_amount = self.total_excretions * compression_ratio
        self.total_excretions *= (1 - compression_ratio)
        
        # Convert some compressed excretions to resources
        resource_recovery = compressed_amount * 0.3
        self.global_resources += resource_recovery
        
        # Affect agent bloom scores based on excretion buffer
        for agent in self.agents:
            if agent.excretion_buffer > 0:
                compression_benefit = min(agent.excretion_buffer * 0.1, 0.05)
                agent.bloom_score *= (1 + compression_benefit)
                agent.excretion_buffer *= 0.5  # Reduce buffer
    
    def apply_mutations(self):
        """Apply small stochastic mutations to agents."""
        mutation_rate = 0.05  # 5% of agents mutate each timestep
        num_mutations = int(len(self.agents) * mutation_rate) + 1
        
        for _ in range(num_mutations):
            if not self.agents:
                break
                
            agent = random.choice(self.agents)
            mutation = ml_suggest_mutation(agent)
            
            # Apply mutations with constraints
            agent.ae += mutation['ae_delta']
            agent.c += mutation['c_delta']
            agent.bloom_score += mutation['bloom_delta']
            
            # Maintain AE = C = 1 constraint (approximately)
            agent.ae = max(0.5, min(1.5, agent.ae))
            agent.c = max(0.5, min(1.5, agent.c))
            agent.bloom_score = max(0.1, min(2.0, agent.bloom_score))
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect simulation metrics for analysis."""
        if not self.agents:
            return {
                'timestep': self.timestep,
                'population': 0,
                'total_excretions': self.total_excretions,
                'avg_bloom': 0.0,
                'avg_ae': 0.0,
                'avg_c': 0.0,
                'resources_available': self.global_resources,
                'spawn_events': self.spawn_events,
                'compression_events': self.compression_events,
                'rby_distribution': [0, 0, 0]
            }
        
        blooms = [a.bloom_score for a in self.agents]
        aes = [a.ae for a in self.agents]
        cs = [a.c for a in self.agents]
        rby_counts = [0, 0, 0]
        
        for agent in self.agents:
            rby_counts[agent.rby_state] += 1
        
        return {
            'timestep': self.timestep,
            'population': len(self.agents),
            'total_excretions': self.total_excretions,
            'avg_bloom': np.mean(blooms),
            'avg_ae': np.mean(aes),
            'avg_c': np.mean(cs),
            'resources_available': self.global_resources,
            'spawn_events': self.spawn_events,
            'compression_events': self.compression_events,
            'rby_distribution': rby_counts
        }


class SimulationVisualizer:
    """Handles visualization of the AIO simulation."""
    
    def __init__(self, state: SimulationState, output_dir: Path, headless: bool = False):
        self.state = state
        self.output_dir = output_dir
        self.headless = headless
        self.metrics_history = []
        
        if not headless:
            # Set up interactive plotting
            plt.ion()
            self.fig, (self.ax_agents, self.ax_metrics) = plt.subplots(1, 2, figsize=(12, 5))
        else:
            # Set up for file output
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            self.fig, (self.ax_agents, self.ax_metrics) = plt.subplots(1, 2, figsize=(12, 5))
    
    def update_visualization(self, metrics: Dict[str, Any]):
        """Update the visualization with current state."""
        self.metrics_history.append(metrics)
        
        # Clear axes
        self.ax_agents.clear()
        self.ax_metrics.clear()
        
        # Plot agents as colored dots
        if self.state.agents:
            positions = np.array([agent.position for agent in self.state.agents])
            colors = [agent.get_color() for agent in self.state.agents]
            sizes = [agent.bloom_score * 50 for agent in self.state.agents]  # Size based on bloom
            
            self.ax_agents.scatter(positions[:, 0], positions[:, 1], 
                                 c=colors, s=sizes, alpha=0.7)
        
        self.ax_agents.set_xlim(0, 1)
        self.ax_agents.set_ylim(0, 1)
        self.ax_agents.set_title(f'Agents (t={metrics["timestep"]}, n={metrics["population"]})')
        self.ax_agents.set_xlabel('X Position')
        self.ax_agents.set_ylabel('Y Position')
        
        # Plot key metrics
        if len(self.metrics_history) > 1:
            timesteps = [m['timestep'] for m in self.metrics_history]
            populations = [m['population'] for m in self.metrics_history]
            resources = [m['resources_available'] for m in self.metrics_history]
            avg_blooms = [m['avg_bloom'] for m in self.metrics_history]
            
            self.ax_metrics.plot(timesteps, populations, 'b-', label='Population')
            self.ax_metrics.plot(timesteps, [r/100 for r in resources], 'g-', label='Resources/100')
            self.ax_metrics.plot(timesteps, avg_blooms, 'r-', label='Avg Bloom')
            
        self.ax_metrics.set_xlabel('Timestep')
        self.ax_metrics.set_ylabel('Value')
        self.ax_metrics.set_title('Key Metrics')
        self.ax_metrics.legend()
        self.ax_metrics.grid(True)
        
        if not self.headless:
            plt.pause(0.01)
    
    def save_final_plots(self):
        """Save final visualization plots."""
        if self.metrics_history:
            # Save current state
            self.fig.savefig(self.output_dir / 'final_state.png', dpi=150, bbox_inches='tight')
            
            # Create detailed metrics plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            timesteps = [m['timestep'] for m in self.metrics_history]
            
            # Population over time
            populations = [m['population'] for m in self.metrics_history]
            axes[0, 0].plot(timesteps, populations)
            axes[0, 0].set_title('Population Over Time')
            axes[0, 0].set_xlabel('Timestep')
            axes[0, 0].set_ylabel('Population')
            axes[0, 0].grid(True)
            
            # Resources and excretions
            resources = [m['resources_available'] for m in self.metrics_history]
            excretions = [m['total_excretions'] for m in self.metrics_history]
            axes[0, 1].plot(timesteps, resources, label='Resources')
            axes[0, 1].plot(timesteps, excretions, label='Excretions')
            axes[0, 1].set_title('Resources and Excretions')
            axes[0, 1].set_xlabel('Timestep')
            axes[0, 1].set_ylabel('Amount')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Average bloom score
            avg_blooms = [m['avg_bloom'] for m in self.metrics_history]
            axes[1, 0].plot(timesteps, avg_blooms)
            axes[1, 0].set_title('Average Bloom Score')
            axes[1, 0].set_xlabel('Timestep')
            axes[1, 0].set_ylabel('Bloom Score')
            axes[1, 0].grid(True)
            
            # AE = C tracking
            avg_aes = [m['avg_ae'] for m in self.metrics_history]
            avg_cs = [m['avg_c'] for m in self.metrics_history]
            axes[1, 1].plot(timesteps, avg_aes, label='Avg AE')
            axes[1, 1].plot(timesteps, avg_cs, label='Avg C')
            axes[1, 1].axhline(y=1.0, color='k', linestyle='--', label='AE = C = 1')
            axes[1, 1].set_title('AE = C = 1 Validation')
            axes[1, 1].set_xlabel('Timestep')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'metrics_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()


def save_metrics_csv(metrics_history: List[Dict[str, Any]], output_dir: Path):
    """Save metrics to CSV for offline analysis."""
    if not metrics_history:
        return
    
    # Flatten RBY distribution for CSV
    flattened_metrics = []
    for metrics in metrics_history:
        flat = metrics.copy()
        rby = flat.pop('rby_distribution')
        flat['rby_red_count'] = rby[0]
        flat['rby_blue_count'] = rby[1]  
        flat['rby_yellow_count'] = rby[2]
        flattened_metrics.append(flat)
    
    df = pd.DataFrame(flattened_metrics)
    csv_path = output_dir / 'simulation_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")


def run_simulation(steps: int, seed: int, max_population: int, 
                  output_dir: Path, headless: bool = False) -> SimulationState:
    """
    Run the AIO simulation with specified parameters.
    
    Args:
        steps: Number of timesteps to simulate
        seed: Random seed for reproducibility
        max_population: Maximum number of agents allowed
        output_dir: Directory for output files
        headless: Run without interactive display
    
    Returns:
        Final simulation state
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Initialize simulation
    state = SimulationState()
    
    # Add initial agents
    initial_population = min(10, max_population)
    for i in range(initial_population):
        agent = Agent(agent_id=i)
        state.add_agent(agent)
    
    # Set up visualization
    visualizer = SimulationVisualizer(state, output_dir, headless)
    
    print(f"Starting simulation: {steps} steps, seed={seed}, initial_pop={initial_population}")
    print(f"Output directory: {output_dir}")
    
    # Run simulation
    start_time = time.time()
    for step in range(steps):
        metrics = state.step()
        
        # Population control
        if len(state.agents) > max_population:
            # Remove oldest agents
            state.agents.sort(key=lambda a: a.age, reverse=True)
            state.agents = state.agents[:max_population]
        
        # Update visualization
        if step % 10 == 0 or step == steps - 1:  # Update every 10 steps
            visualizer.update_visualization(metrics)
            
        # Progress reporting
        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}/{steps} - Pop: {metrics['population']}, "
                  f"Resources: {metrics['resources_available']:.1f}, "
                  f"Bloom: {metrics['avg_bloom']:.3f}, "
                  f"Spawns: {metrics['spawn_events']}, "
                  f"Compressions: {metrics['compression_events']} "
                  f"({elapsed:.1f}s)")
    
    # Save results
    visualizer.save_final_plots()
    save_metrics_csv(visualizer.metrics_history, output_dir)
    
    # Save final state summary
    final_metrics = state.collect_metrics()
    summary = {
        'simulation_parameters': {
            'steps': steps,
            'seed': seed,
            'max_population': max_population,
            'initial_population': initial_population
        },
        'final_state': final_metrics,
        'total_runtime_seconds': time.time() - start_time
    }
    
    with open(output_dir / 'simulation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSimulation complete! Final population: {final_metrics['population']}")
    print(f"AE convergence: {final_metrics['avg_ae']:.3f} ≈ 1.0")
    print(f"C convergence: {final_metrics['avg_c']:.3f} ≈ 1.0") 
    print(f"Total spawn events: {final_metrics['spawn_events']}")
    print(f"Total compression events: {final_metrics['compression_events']}")
    
    return state


def main():
    """Main entry point for the AIO simulation."""
    parser = argparse.ArgumentParser(
        description='AIOS IO Simulation - AE=C=1 and Law of Three',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--steps', type=int, default=500,
                       help='Number of simulation timesteps (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--max-population', type=int, default=100,
                       help='Maximum agent population (default: 100)')
    parser.add_argument('--outdir', type=str, default='output',
                       help='Output directory for results (default: output)')
    parser.add_argument('--headless', action='store_true',
                       help='Run without interactive display')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for display availability
    headless = args.headless or not os.environ.get('DISPLAY')
    if headless and not args.headless:
        print("No DISPLAY detected, running in headless mode")
    
    try:
        # Run simulation
        final_state = run_simulation(
            steps=args.steps,
            seed=args.seed,
            max_population=args.max_population,
            output_dir=output_dir,
            headless=headless
        )
        
        print(f"\nResults saved to: {output_dir.absolute()}")
        print("Files created:")
        for file_path in output_dir.iterdir():
            print(f"  - {file_path.name}")
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nSimulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()