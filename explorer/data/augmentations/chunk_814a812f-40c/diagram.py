import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon
import matplotlib.lines as mlines

# Define Fano plane coordinates for visual clarity
def get_fano_coordinates():
    # 7 points arranged in a triangle with center
    coords = {
        0: np.array([0, 1]),          # top
        1: np.array([-0.866, -0.5]),  # bottom left
        2: np.array([0.866, -0.5]),   # bottom right
        3: np.array([0, 0]),          # center
        4: np.array([0, -0.5]),       # bottom middle
        5: np.array([-0.433, 0.25]),  # left middle
        6: np.array([0.433, 0.25]),   # right middle
    }
    return coords

# Define the 7 lines of Fano plane
def get_fano_lines():
    return [
        [0, 1, 4],  # Line 0
        [1, 2, 5],  # Line 1
        [2, 0, 6],  # Line 2
        [3, 4, 5],  # Line 3
        [3, 6, 1],  # Line 4
        [3, 0, 2],  # Line 5
        [4, 6, 5],  # Line 6 (inscribed circle)
    ]

def draw_fano_base(ax, coords, highlight_points=None, highlight_lines=None):
    # Draw all points
    for i, coord in coords.items():
        color = 'red' if highlight_points and i in highlight_points else 'black'
        size = 150 if highlight_points and i in highlight_points else 100
        ax.scatter(*coord, c=color, s=size, zorder=5)
        ax.text(coord[0]*1.2, coord[1]*1.2, str(i), fontsize=10, ha='center', va='center')
    
    # Draw all lines
    lines = get_fano_lines()
    for i, line in enumerate(lines):
        if i < 6:  # Straight lines
            for j in range(len(line)):
                p1, p2 = coords[line[j]], coords[line[(j+1)%len(line)]]
                is_highlight = highlight_lines and i in highlight_lines
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                       'b-' if is_highlight else 'gray', 
                       linewidth=3 if is_highlight else 1,
                       alpha=1 if is_highlight else 0.5)
        else:  # Inscribed circle (line 6)
            circle = Circle((0, 0), 0.5, fill=False, 
                          color='b' if highlight_lines and i in highlight_lines else 'gray',
                          linewidth=3 if highlight_lines and i in highlight_lines else 1,
                          alpha=1 if highlight_lines and i in highlight_lines else 0.5)
            ax.add_patch(circle)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Get coordinates
coords = get_fano_coordinates()

# Part 1: Antiflag (point 3, line 2)
ax1.set_title('Antiflag Orbit: Point-Opposite Line\n(28 elements)', fontsize=12, fontweight='bold')
draw_fano_base(ax1, coords, highlight_points=[3], highlight_lines=[2])

# Add labels for the antiflag
ax1.text(0, -1.5, 'p = 3 (consciousness point)', color='red', fontsize=10, ha='center')
ax1.text(0, -1.7, 'L = {0, 2, 6} (opposite line)', color='blue', fontsize=10, ha='center')

# Part 2: Point-Triangle Incidence (point 0, triangle {1,2,3})
ax2.set_title('Incidence Orbit: Point in Triangle\n(84 elements)', fontsize=12, fontweight='bold')

# Draw base Fano
draw_fano_base(ax2, coords, highlight_points=[0])

# Highlight triangle
triangle_points = [1, 2, 3]
triangle_coords = [coords[p] for p in triangle_points]
triangle = Polygon(triangle_coords, fill=False, edgecolor='green', linewidth=3)
ax2.add_patch(triangle)

# Fill triangle lightly
triangle_fill = Polygon(triangle_coords, fill=True, facecolor='green', alpha=0.1)
ax2.add_patch(triangle_fill)

# Add labels
ax2.text(0, -1.5, 'P = 0 (participating point)', color='red', fontsize=10, ha='center')
ax2.text(0, -1.7, 'T = {1, 2, 3} (trinity triangle)', color='green', fontsize=10, ha='center')

# Format both axes
for ax in [ax1, ax2]:
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

# Add main title
fig.suptitle('The 112 Chakras: Two Aut(Fano) Orbits', fontsize=14, fontweight='bold')

# Add legend
legend_elements = [
    mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, label='Highlighted point'),
    mlines.Line2D([], [], color='blue', linewidth=3, label='Opposite line (antiflag)'),
    mlines.Line2D([], [], color='green', linewidth=3, label='Triangle (incidence)')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.savefig('fano_chakra_orbits.png', dpi=300, bbox_inches='tight')
plt.show()