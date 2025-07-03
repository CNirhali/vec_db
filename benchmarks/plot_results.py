import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_CSV = 'benchmarks/results/benchmark_results.csv'
PLOT_DIR = 'benchmarks/results/'

def plot_benchmarks():
    df = pd.read_csv(RESULTS_CSV)
    
    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # Create a comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Y_use_FAISS vs FAISS Benchmark Results', fontsize=16, fontweight='bold')
    
    # 1. Add Time Comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(df)), df['add_time'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    ax1.set_title('Add Time (seconds)', fontweight='bold')
    ax1.set_ylabel('Time (s)')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels([m.replace('VectorDB_', '').replace('_memonlyTrue', '').replace('_memonlyFalse', '') for m in df['method']], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars1, df['add_time']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.2f}s', ha='center', va='bottom', fontsize=8)
    
    # 2. Search Time Comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(df)), df['search_time'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    ax2.set_title('Search Time (seconds)', fontweight='bold')
    ax2.set_ylabel('Time (s)')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels([m.replace('VectorDB_', '').replace('_memonlyTrue', '').replace('_memonlyFalse', '') for m in df['method']], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars2, df['search_time']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}s', ha='center', va='bottom', fontsize=8)
    
    # 3. Memory Usage
    ax3 = axes[0, 2]
    bars3 = ax3.bar(range(len(df)), df['memory'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    ax3.set_title('Memory Usage (MB)', fontweight='bold')
    ax3.set_ylabel('Memory (MB)')
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels([m.replace('VectorDB_', '').replace('_memonlyTrue', '').replace('_memonlyFalse', '') for m in df['method']], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars3, df['memory']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.2f}MB', ha='center', va='bottom', fontsize=8)
    
    # 4. Recall Comparison
    ax4 = axes[1, 0]
    bars4 = ax4.bar(range(len(df)), df['recall'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    ax4.set_title('Recall@10', fontweight='bold')
    ax4.set_ylabel('Recall')
    ax4.set_ylim(0, 1.1)
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels([m.replace('VectorDB_', '').replace('_memonlyTrue', '').replace('_memonlyFalse', '') for m in df['method']], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars4, df['recall']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 5. Speed-Recall Tradeoff (Search Time vs Recall)
    ax5 = axes[1, 1]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, (method, color) in enumerate(zip(df['method'], colors)):
        label = method.replace('VectorDB_', '').replace('_memonlyTrue', '').replace('_memonlyFalse', '')
        ax5.scatter(df.iloc[i]['search_time'], df.iloc[i]['recall'], 
                   c=color, s=100, alpha=0.7, label=label)
        ax5.annotate(label, (df.iloc[i]['search_time'], df.iloc[i]['recall']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax5.set_xlabel('Search Time (seconds)')
    ax5.set_ylabel('Recall@10')
    ax5.set_title('Speed-Recall Tradeoff', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Speed-Recall Tradeoff (Add Time vs Recall)
    ax6 = axes[1, 2]
    for i, (method, color) in enumerate(zip(df['method'], colors)):
        label = method.replace('VectorDB_', '').replace('_memonlyTrue', '').replace('_memonlyFalse', '')
        ax6.scatter(df.iloc[i]['add_time'], df.iloc[i]['recall'], 
                   c=color, s=100, alpha=0.7, label=label)
        ax6.annotate(label, (df.iloc[i]['add_time'], df.iloc[i]['recall']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax6.set_xlabel('Add Time (seconds)')
    ax6.set_ylabel('Recall@10')
    ax6.set_title('Add Time vs Recall Tradeoff', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'comprehensive_benchmark.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual plots for each metric
    metrics = ['add_time', 'search_time', 'memory', 'recall']
    metric_names = ['Add Time (seconds)', 'Search Time (seconds)', 'Memory Usage (MB)', 'Recall@10']
    
    for metric, metric_name in zip(metrics, metric_names):
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(df)), df[metric], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
        plt.title(f'Benchmark: {metric_name}', fontweight='bold', fontsize=14)
        plt.ylabel(metric_name)
        plt.xticks(range(len(df)), [m.replace('VectorDB_', '').replace('_memonlyTrue', '').replace('_memonlyFalse', '') for m in df['method']], rotation=45, ha='right')
        
        # Add value labels
        for bar, value in zip(bars, df[metric]):
            height = bar.get_height()
            if metric == 'recall':
                plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            elif metric == 'memory':
                plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.2f}MB', ha='center', va='bottom', fontsize=10)
            else:
                plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.3f}s', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f'{metric}_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"[RESULT] Comprehensive plot saved to {PLOT_DIR}comprehensive_benchmark.png")
    print(f"[RESULT] Individual plots saved to {PLOT_DIR}")

if __name__ == "__main__":
    plot_benchmarks() 