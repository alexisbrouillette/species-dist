import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec

def visualize_training_history(results_df, scenario_indices=None, metrics=['loss', 'f1', 'pr_auc']):
    """
    Plot training history for scenarios in the results DataFrame
    
    Args:
        results_df: DataFrame with training results (each row is a scenario)
        scenario_indices: List of row indices to plot (None for all)
        metrics: List of metrics to plot
    """
    if scenario_indices is None:
        scenario_indices = range(len(results_df))
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenario_indices)))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for j, idx in enumerate(scenario_indices):
            if idx >= len(results_df):
                continue
                
            row = results_df.iloc[idx]
            scenario_name = f"{row.get('config_name', f'Config_{idx}')}_{row.get('cell_size', '')}"
            color = colors[j]
            
            # Plot training metric
            if 'train_metrics' in row and metric in row['train_metrics']:
                train_metric = row['train_metrics'][metric]
                epochs = range(1, len(train_metric) + 1)
                ax.plot(epochs, train_metric, '--', color=color, alpha=0.7, label=f'{scenario_name} (Train)')
            
            # Plot validation metric
            if 'val_metrics' in row and metric in row['val_metrics']:
                val_metric = row['val_metrics'][metric]
                epochs = range(1, len(val_metric) + 1)
                ax.plot(epochs, val_metric, '-', color=color, alpha=1, label=f'{scenario_name} (Val)')
                
                # Mark best epoch
                best_epoch = row.get('best_epoch', 0)
                if best_epoch > 0 and best_epoch <= len(val_metric):
                    best_val = val_metric[best_epoch-1]
                    ax.scatter(best_epoch, best_val, color=color, s=100, zorder=5)
        
        ax.set_title(metric.upper())
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_species_metrics(results_df, scenario_indices=None):
    """
    Plot per-species metrics for scenarios in the results DataFrame
    
    Args:
        results_df: DataFrame with training results
        scenario_indices: List of row indices to plot (None for all)
    """
    if scenario_indices is None:
        scenario_indices = range(len(results_df))
    
    # Get species list from first valid scenario
    species_list = []
    for idx in scenario_indices:
        if idx < len(results_df):
            row = results_df.iloc[idx]
            if 'final_val_metrics' in row and 'per_species' in row['final_val_metrics']:
                species_list = list(row['final_val_metrics']['per_species'].keys())
                break
    
    if not species_list:
        print("No species metrics found")
        return
    
    metrics = ['f1', 'precision', 'recall', 'pr_auc']
    n_species = len(species_list)
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(n_species, n_metrics, figsize=(5*n_metrics, 3*n_species))
    if n_species == 1:
        axes = [axes]
    if n_metrics == 1:
        for i in range(n_species):
            axes[i] = [axes[i]]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenario_indices)))
    
    for i, species in enumerate(species_list):
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            
            values = []
            scenario_names = []
            
            for k, idx in enumerate(scenario_indices):
                if idx >= len(results_df):
                    continue
                    
                row = results_df.iloc[idx]
                scenario_name = f"{row.get('config_name', f'Config_{idx}')}_{row.get('cell_size', '')}"
                
                if ('final_val_metrics' in row and 
                    'per_species' in row['final_val_metrics'] and
                    species in row['final_val_metrics']['per_species'] and
                    metric in row['final_val_metrics']['per_species'][species]):
                    
                    value = row['final_val_metrics']['per_species'][species][metric]
                    values.append(value)
                    scenario_names.append(scenario_name)
            
            if values:
                x_pos = np.arange(len(values))
                bars = ax.bar(x_pos, values, color=colors[:len(values)], alpha=0.7)
                
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{value:.3f}', ha='center', va='bottom')
            
            ax.set_title(f'{species} - {metric.upper()}')
            ax.set_ylabel(metric)
            ax.set_xticks(x_pos if values else [])
            ax.set_xticklabels(scenario_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

def compare_scenarios(results_df, scenario_indices=None):
    """
    Create comprehensive comparison of scenarios in the results DataFrame
    
    Args:
        results_df: DataFrame with training results
        scenario_indices: List of row indices to compare (None for all)
    """
    if scenario_indices is None:
        scenario_indices = range(len(results_df))
    
    comparison_data = []
    scenario_names = []
    
    for idx in scenario_indices:
        if idx >= len(results_df):
            continue
            
        row = results_df.iloc[idx]
        scenario_name = f"{row.get('config_name', f'Config_{idx}')}_{row.get('cell_size', '')}"
        
        if 'final_val_metrics' in row and 'macro' in row['final_val_metrics']:
            macro_metrics = row['final_val_metrics']['macro'].copy()
            row_data = {
                'Scenario': scenario_name,
                'Config': row.get('config_name', 'N/A'),
                'Cell_Size': row.get('cell_size', 'N/A'),
                'Species': row.get('species', 'N/A'),
                'Epochs': row.get('epochs_completed', 'N/A'),
                'Stopped_Early': row.get('stopped_early', 'N/A'),
                'Best_Epoch': row.get('best_epoch', 'N/A')
            }
            row_data.update(macro_metrics)
            comparison_data.append(row_data)
            scenario_names.append(scenario_name)
    
    if not comparison_data:
        print("No valid data to compare")
        return pd.DataFrame()
    
    df = pd.DataFrame(comparison_data)
    
    # Plot macro metrics comparison
    metrics_to_plot = [m for m in ['loss', 'accuracy', 'f1', 'pr_auc'] if m in df.columns]
    n_metrics = len(metrics_to_plot)
    
    if n_metrics > 0:
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(comparison_data)))
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            values = df[metric].values
            x_pos = np.arange(len(values))
            bars = ax.bar(x_pos, values, color=colors, alpha=0.7)
            
            ax.set_title(metric.upper())
            ax.set_ylabel(metric)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(scenario_names, rotation=45, ha='right')
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    return df

def plot_metric_comparison(results_df, metric='f1', scenario_indices=None):
    """
    Plot comparison of a specific metric across scenarios
    
    Args:
        results_df: DataFrame with training results
        metric: Metric to compare ('loss', 'f1', 'pr_auc', etc.)
        scenario_indices: List of row indices to compare (None for all)
    """
    if scenario_indices is None:
        scenario_indices = range(len(results_df))
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenario_indices)))
    
    for i, idx in enumerate(scenario_indices):
        if idx >= len(results_df):
            continue
            
        row = results_df.iloc[idx]
        scenario_name = f"{row.get('config_name', f'Config_{idx}')}_{row.get('cell_size', '')}"
        
        if 'val_metrics' in row and metric in row['val_metrics']:
            val_metric = row['val_metrics'][metric]
            epochs = range(1, len(val_metric) + 1)
            plt.plot(epochs, val_metric, '-', color=colors[i], label=scenario_name, linewidth=2)
            
            # Mark best epoch
            best_epoch = row.get('best_epoch', 0)
            if best_epoch > 0 and best_epoch <= len(val_metric):
                best_val = val_metric[best_epoch-1]
                plt.scatter(best_epoch, best_val, color=colors[i], s=100, zorder=5)
    
    plt.xlabel('Epoch')
    plt.ylabel(metric.upper())
    plt.title(f'{metric.upper()} Comparison Across Scenarios')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def get_scenario_details(results_df, index):
    """
    Get detailed information about a specific scenario
    
    Args:
        results_df: DataFrame with training results
        index: Row index of the scenario
    """
    if index >= len(results_df):
        print(f"Index {index} out of bounds")
        return
    
    row = results_df.iloc[index]
    print("=" * 60)
    print(f"SCENARIO DETAILS - Index {index}")
    print("=" * 60)
    print(f"Config: {row.get('config_name', 'N/A')}")
    print(f"Cell Size: {row.get('cell_size', 'N/A')}")
    print(f"Species: {row.get('species', 'N/A')}")
    print(f"Epochs Completed: {row.get('epochs_completed', 'N/A')}")
    print(f"Stopped Early: {row.get('stopped_early', 'N/A')}")
    print(f"Best Epoch: {row.get('best_epoch', 'N/A')}")
    print(f"Best Val Metric: {row.get('best_val_metric', 'N/A'):.4f}")
    
    if 'final_val_metrics' in row and 'macro' in row['final_val_metrics']:
        print("\nFinal Validation Metrics (Macro):")
        for metric, value in row['final_val_metrics']['macro'].items():
            print(f"  {metric}: {value:.4f}")
    
    print("\nTraining Positive Ratio:", row.get('train_positive_ratio', 'N/A'))
    print("Validation Positive Ratio:", row.get('val_positive_ratio', 'N/A'))
    print("=" * 60)

# Example usage:
def demonstrate_visualizations(results_df):
    """
    Demonstrate all visualization functions
    """
    print("Results DataFrame shape:", results_df.shape)
    print("Available columns:", results_df.columns.tolist())
    
    results_df['config_name'] = results_df['config'].apply(lambda c: c.get('name', 'N/A') if isinstance(c, dict) else 'N/A')
    # Show basic info about each scenario
    for i in range(min(5, len(results_df))):
        get_scenario_details(results_df, i)
    
    # Plot training history for all scenarios
    visualize_training_history(results_df)
    
    # Plot species metrics for all scenarios
    #visualize_species_metrics(results_df)
    
    # Compare scenarios and get comparison table
    comparison_df = compare_scenarios(results_df)
    print("\nComparison Table:")
    print(comparison_df.to_string(index=False))
    
    # Plot specific metric comparison
    plot_metric_comparison(results_df, metric='f1')
    plot_metric_comparison(results_df, metric='pr_auc')
    
    # Compare specific scenarios
    if len(results_df) >= 2:
        visualize_training_history(results_df, [0, 1], metrics=['loss', 'f1'])
        compare_scenarios(results_df, [0, 1])