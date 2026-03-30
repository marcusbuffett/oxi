import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

results = [
    {'iter': 0, 'ao10': 0.330795, 'status': 'baseline'},
    {'iter': 1, 'ao10': 0.337110, 'status': 'kept'},
    {'iter': 2, 'ao10': 0.336623, 'status': 'discarded'},
    {'iter': 3, 'ao10': 0.328016, 'status': 'discarded'},
    {'iter': 4, 'ao10': 0.345991, 'status': 'kept'},
    {'iter': 5, 'ao10': 0.329589, 'status': 'discarded'},
    {'iter': 6, 'ao10': 0.337002, 'status': 'discarded'},
    {'iter': 7, 'ao10': 0.331107, 'status': 'discarded'},
    {'iter': 8, 'ao10': 0.337010, 'status': 'discarded'},
    {'iter': 9, 'ao10': 0.329562, 'status': 'discarded'},
    {'iter': 10, 'ao10': 0.339916, 'status': 'discarded'},
    {'iter': 11, 'ao10': 0.331467, 'status': 'discarded'},
    {'iter': 12, 'ao10': 0.350539, 'status': 'kept'},
    {'iter': 13, 'ao10': 0.328959, 'status': 'discarded'},
    {'iter': 14, 'ao10': 0.336035, 'status': 'discarded'},
    {'iter': 15, 'ao10': 0.339274, 'status': 'discarded'},
    {'iter': 16, 'ao10': 0.333685, 'status': 'discarded'},
    {'iter': 17, 'ao10': 0.338022, 'status': 'discarded'},
    {'iter': 18, 'ao10': 0.331056, 'status': 'discarded'},
    {'iter': 19, 'ao10': 0.330982, 'status': 'discarded'},
    {'iter': 20, 'ao10': 0.339065, 'status': 'discarded'},
]

baseline = 0.330795

fig, ax = plt.subplots(figsize=(14, 7))

for r in results:
    if r['status'] == 'kept':
        color = '#2ecc71'
        marker = '^'
        size = 120
    elif r['status'] == 'baseline':
        color = '#3498db'
        marker = 'D'
        size = 120
    else:
        color = '#e74c3c'
        marker = 'v'
        size = 80
    ax.scatter(r['iter'], r['ao10'], c=color, marker=marker, s=size, zorder=3, edgecolors='black', linewidth=0.5)

ax.axhline(y=baseline, color='#3498db', linestyle='--', alpha=0.5, linewidth=1.5, label=f'Baseline: {baseline:.6f}')

# Best-so-far step line
best_so_far = baseline
best_x = [0]
best_y = [baseline]
for r in results:
    if r['ao10'] > best_so_far and r['status'] in ('kept', 'baseline'):
        best_so_far = r['ao10']
        best_x.append(r['iter'])
        best_y.append(best_so_far)
best_x.append(20)
best_y.append(best_so_far)
ax.step(best_x, best_y, where='post', color='#2ecc71', alpha=0.6, linewidth=2.5, label=f'Best: {best_so_far:.6f}')

# Labels for kept iterations
for r in results:
    if r['status'] == 'kept':
        ax.annotate(f"iter {r['iter']}\n{r['ao10']:.4f}", 
                    xy=(r['iter'], r['ao10']), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=8, fontweight='bold', color='#27ae60',
                    arrowprops=dict(arrowstyle='->', color='#27ae60', lw=0.8))

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('ao10 Accuracy', fontsize=12)
ax.set_title('Oxi Chess Model Research Loop — 20 Iterations', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, 20.5)

# Custom legend entries
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#2ecc71', markersize=10, label='Kept (improvement)'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='#e74c3c', markersize=10, label='Discarded'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#3498db', markersize=10, label='Baseline'),
    Line2D([0], [0], color='#3498db', linestyle='--', alpha=0.5, linewidth=1.5, label=f'Baseline: {baseline:.6f}'),
    Line2D([0], [0], color='#2ecc71', alpha=0.6, linewidth=2.5, label=f'Best: {best_so_far:.6f}'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='lower right')

plt.tight_layout()
chart_path = Path("/Users/marcusbuffett/projects/chessbook/oxi/research_runs/progress.png")
chart_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(chart_path, dpi=150)
plt.close()
print(f"Chart saved to {chart_path}")
