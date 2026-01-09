import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, FancyBboxPatch
from matplotlib.lines import Line2D

# Set up the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Define the 7 points of Fano plane in a symmetric arrangement
theta = np.linspace(0, 2*np.pi, 7, endpoint=False)
radius = 2
points = {
    'P0': (0, 0),  # center point
    'P1': (radius * np.cos(theta[0]), radius * np.sin(theta[0])),
    'P2': (radius * np.cos(theta[1]), radius * np.sin(theta[1])),
    'P3': (radius * np.cos(theta[2]), radius * np.sin(theta[2])),
    'P4': (radius * np.cos(theta[3]), radius * np.sin(theta[3])),
    'P5': (radius * np.cos(theta[4]), radius * np.sin(theta[4])),
    'P6': (radius * np.cos(theta[5]), radius * np.sin(theta[5]))
}

# Define the 7 lines of Fano plane (each line contains exactly 3 points)
lines = {
    'L1': ['P1', 'P2', 'P4'],
    'L2': ['P2', 'P3', 'P5'],
    'L3': ['P3', 'P4', 'P6'],
    'L4': ['P4', 'P5', 'P1'],
    'L5': ['P5', 'P6', 'P2'],
    'L6': ['P6', 'P1', 'P3'],
    'L7': ['P0', 'P1', 'P5'],  # Lines through center
    'L∞': ['P0', 'P2', 'P6'],  # Earth horizon line
    'L9': ['P0', 'P3', 'P4']
}

# Set C (Heart point) as P1 and L∞ as the line containing P0, P2, P6
C = 'P1'
L_infinity = 'L∞'
body_points = [p for p in points.keys() if p != C and p not in lines[L_infinity]]

# Function to draw Fano plane
def draw_fano(ax, highlight_C=True, highlight_L_inf=True):
    # Draw points
    for name, pos in points.items():
        if name == C and highlight_C:
            ax.scatter(*pos, s=300, c='red', marker='*', zorder=5, edgecolor='darkred', linewidth=2)
            ax.text(pos[0], pos[1]+0.3, 'C (Heart)', ha='center', fontsize=10, fontweight='bold', color='red')
        elif name in lines[L_infinity] and highlight_L_inf:
            ax.scatter(*pos, s=200, c='brown', marker='s', zorder=4, edgecolor='black', linewidth=1)
        else:
            ax.scatter(*pos, s=150, c='lightblue', edgecolor='navy', linewidth=1, zorder=3)
        
        if name != C:
            ax.text(pos[0]*1.15, pos[1]*1.15, name, ha='center', va='center', fontsize=9)
    
    # Draw lines
    for line_name, line_points in lines.items():
        if len(line_points) == 3:
            if line_name == L_infinity and highlight_L_inf:
                # Draw Earth horizon line specially
                p1, p2, p3 = [points[p] for p in line_points]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'brown', linewidth=3, alpha=0.7)
                ax.plot([p2[0], p3[0]], [p2[1], p3[1]], 'brown', linewidth=3, alpha=0.7)
                ax.plot([p3[0], p1[0]], [p3[1], p1[1]], 'brown', linewidth=3, alpha=0.7)
                ax.text(0, -3.2, 'L∞ (Earth Horizon)', ha='center', fontsize=11, 
                       color='brown', fontweight='bold')
            else:
                # Draw other lines
                p1, p2, p3 = [points[p] for p in line_points]
                color = 'gray'
                alpha = 0.3
                if C in line_points:
                    color = 'orange'
                    alpha = 0.5
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color, linewidth=1.5, alpha=alpha)
                ax.plot([p2[0], p3[0]], [p2[1], p3[1]], color, linewidth=1.5, alpha=alpha)
                ax.plot([p3[0], p1[0]], [p3[1], p1[1]], color, linewidth=1.5, alpha=alpha)

# Left panel: Fano plane with C and L∞ highlighted
ax1.set_title('Fano Plane with Fixed Heart Point C and Earth Horizon L∞', fontsize=14, fontweight='bold')
draw_fano(ax1)
ax1.set_xlim(-3.5, 3.5)
ax1.set_ylim(-3.5, 3.5)
ax1.set_aspect('equal')
ax1.axis('off')

# Right panel: Quadrangle type classification
ax2.set_title('Quadrangle Classification by Affine Density', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Define quadrangle types with colors
quad_types = {
    'Type I (4 body pts)': {'color': '#FFE4B5', 'count': 12, 'y': 8.5},
    'Type IIa (3 body pts)': {'color': '#98FB98', 'count': 12, 'y': 7},
    'Type IIb (3 body pts)': {'color': '#87CEEB', 'count': 12, 'y': 5.5},
    'Type IIIa (2 body pts)': {'color': '#DDA0DD', 'count': 12, 'y': 4},
    'Type IIIb (2 body pts)': {'color': '#F0E68C', 'count': 12, 'y': 2.5},
    'Type IIIc (2 body pts)': {'color': '#FFB6C1', 'count': 12, 'y': 1}
}

# Draw boxes for each type
box_width = 7
box_height = 0.8
for quad_type, props in quad_types.items():
    box = FancyBboxPatch((1, props['y']-box_height/2), box_width, box_height,
                         boxstyle="round,pad=0.05", 
                         facecolor=props['color'],
                         edgecolor='black',
                         linewidth=1.5)
    ax2.add_patch(box)
    ax2.text(1.5, props['y'], quad_type, va='center', fontsize=11, fontweight='bold')
    ax2.text(7.5, props['y'], f"{props['count']} quadrangles", va='center', fontsize=10)

# Add totals
ax2.text(5, 9.5, 'Total: 28×4 = 112 chakras', ha='center', fontsize=12, fontweight='bold')
ax2.text(5, 0.2, 'Partition: 12+72+4+6+6+12 = 112', ha='center', fontsize=12, 
         fontweight='bold', color='darkgreen')

# Add legend for left panel
legend_elements = [
    Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, 
           label='C (Heart point)', markeredgecolor='darkred'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='brown', markersize=10, 
           label='Points on L∞', markeredgecolor='black'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, 
           label='Body points', markeredgecolor='navy'),
    Line2D([0], [0], color='brown', linewidth=3, label='L∞ (Earth horizon)'),
    Line2D([0], [0], color='orange', linewidth=2, label='Lines through C'),
    Line2D([0], [0], color='gray', linewidth=1.5, label='Other lines')
]
ax1.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0, 0))

# Add note about affine density
ax2.text(5, -0.5, 'Affine density = number of quadrangle vertices not on C or L∞', 
         ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('fano_chakra_partition.png', dpi=300, bbox_inches='tight')
plt.show()