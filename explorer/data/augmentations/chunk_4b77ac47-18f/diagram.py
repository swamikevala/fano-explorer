import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Define colors
color_total = '#2c3e50'
color_radical = '#e74c3c'
color_bulk = '#3498db'
color_self = '#f39c12'
color_field = '#9b59b6'
color_immanent = '#27ae60'
color_transcendent = '#8e44ad'
color_chiral_left = '#16a085'
color_chiral_right = '#d35400'

# Define positions
y_levels = [8, 6, 4, 2]
x_center = 7

# Level 0: Total 112
rect_total = FancyBboxPatch((x_center-1.5, y_levels[0]-0.3), 3, 0.6,
                            boxstyle="round,pad=0.1",
                            facecolor=color_total, edgecolor='black', linewidth=2)
ax.add_patch(rect_total)
ax.text(x_center, y_levels[0], '112 anti-flags', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='white')

# Level 1: Primary partition (4-72-12-24)
partitions_1 = [
    (1.5, 2, color_radical, '4\nradical\nwitness'),
    (4.5, 4, color_bulk, '72\nbulk\nflow'),
    (8.5, 2, color_self, '12\nself\nflow'),
    (11.5, 3, color_field, '24\nfield\nwitness')
]

for x, width, color, label in partitions_1:
    rect = FancyBboxPatch((x, y_levels[1]-0.4), width, 0.8,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + width/2, y_levels[1], label, ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

# Draw connections from total to primary partition
for x, width, _, _ in partitions_1:
    ax.plot([x_center, x + width/2], [y_levels[0]-0.3, y_levels[1]+0.4],
            'k-', linewidth=1.5, alpha=0.7)

# Level 2: Split 24 into 12+12
partitions_2 = [
    (10.5, 1.5, color_immanent, '12\nimmanent'),
    (12.5, 1.5, color_transcendent, '12\ntranscendent')
]

for x, width, color, label in partitions_2:
    rect = FancyBboxPatch((x, y_levels[2]-0.3), width, 0.6,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + width/2, y_levels[2], label, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

# Connect 24 to 12+12
ax.plot([13, 11.25], [y_levels[1]-0.4, y_levels[2]+0.3], 'k-', linewidth=1.5, alpha=0.7)
ax.plot([13, 13.25], [y_levels[1]-0.4, y_levels[2]+0.3], 'k-', linewidth=1.5, alpha=0.7)

# Level 3: Split transcendent 12 into 6+6
partitions_3 = [
    (12.0, 0.8, color_chiral_left, '6\nleft'),
    (13.2, 0.8, color_chiral_right, '6\nright')
]

for x, width, color, label in partitions_3:
    rect = FancyBboxPatch((x, y_levels[3]-0.25), width, 0.5,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + width/2, y_levels[3], label, ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')

# Connect transcendent 12 to 6+6
ax.plot([13.25, 12.4], [y_levels[2]-0.3, y_levels[3]+0.25], 'k-', linewidth=1.5, alpha=0.7)
ax.plot([13.25, 13.6], [y_levels[2]-0.3, y_levels[3]+0.25], 'k-', linewidth=1.5, alpha=0.7)

# Add labels for splitting criteria
ax.text(x_center, (y_levels[0]+y_levels[1])/2 - 0.3, 'Fix self point P₀', 
        ha='center', va='center', fontsize=10, style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

ax.text(12.5, (y_levels[1]+y_levels[2])/2 + 0.2, 'P₀ membership', 
        ha='center', va='center', fontsize=9, style='italic',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray'))

ax.text(12.5, (y_levels[2]+y_levels[3])/2 + 0.2, 'A₄ orbits\n(chirality)', 
        ha='center', va='center', fontsize=9, style='italic',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray'))

# Add title
ax.text(x_center, 9.5, 'Partition Cascade from Fixing Self Point P₀', 
        ha='center', va='center', fontsize=16, fontweight='bold')

# Add legend
legend_elements = [
    patches.Patch(color=color_radical, label='Radical witness (P₀ ∈ radical center)'),
    patches.Patch(color=color_bulk, label='Bulk flow (generic position)'),
    patches.Patch(color=color_self, label='Self flow (P₀ on self-polar line)'),
    patches.Patch(color=color_field, label='Field witness (P₀ ∈ orthogonal line)'),
]

ax.legend(handles=legend_elements, loc='lower center', ncol=2, 
          frameon=True, fancybox=True, shadow=True, fontsize=10)

# Set axis properties
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

plt.tight_layout()
plt.savefig('partition_cascade.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('partition_cascade.svg', format='svg', bbox_inches='tight', facecolor='white')
plt.show()