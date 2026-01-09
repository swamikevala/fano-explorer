import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon
import matplotlib.lines as mlines

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_aspect('equal')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.axis('off')

# Define the 7 points of the Fano plane in a symmetric arrangement
# p₀ at center, others in hexagon
angles = np.linspace(0, 2*np.pi, 7)[:-1]
radius = 1.5
points = {
    'p₀': (0, 0),
    'p₁': (radius * np.cos(angles[0]), radius * np.sin(angles[0])),
    'p₂': (radius * np.cos(angles[1]), radius * np.sin(angles[1])),
    'p₃': (radius * np.cos(angles[2]), radius * np.sin(angles[2])),
    'p₄': (radius * np.cos(angles[3]), radius * np.sin(angles[3])),
    'p₅': (radius * np.cos(angles[4]), radius * np.sin(angles[4])),
    'p₆': (radius * np.cos(angles[5]), radius * np.sin(angles[5]))
}

# Define the 7 lines of the Fano plane
# Lines through p₀ (shown in blue)
lines_through_p0 = [
    ['p₀', 'p₁', 'p₄'],  # Line 1
    ['p₀', 'p₂', 'p₅'],  # Line 2
    ['p₀', 'p₃', 'p₆']   # Line 3
]

# Lines NOT through p₀ (shown in red) - the Muladhara 4
lines_not_through_p0 = [
    ['p₁', 'p₂', 'p₃'],  # Line 4
    ['p₁', 'p₅', 'p₆'],  # Line 5
    ['p₂', 'p₄', 'p₆'],  # Line 6
    ['p₃', 'p₄', 'p₅']   # Line 7 (circular)
]

# Draw points
for name, pos in points.items():
    if name == 'p₀':
        # Highlight p₀ with a larger, filled circle
        circle = Circle(pos, 0.15, color='black', fill=True, zorder=3)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1]-0.35, name, ha='center', va='top', fontsize=14, 
                fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    else:
        circle = Circle(pos, 0.08, color='black', fill=True, zorder=3)
        ax.add_patch(circle)
        ax.text(pos[0]*1.2, pos[1]*1.2, name, ha='center', va='center', fontsize=12)

# Draw lines through p₀ (blue)
for line in lines_through_p0:
    if len(line) == 3:
        # Draw straight lines
        p1, p2 = points[line[0]], points[line[2]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=2, alpha=0.7, zorder=1)

# Draw lines NOT through p₀ (red) - the Muladhara 4
for i, line in enumerate(lines_not_through_p0):
    if i < 3:  # First three are straight lines
        p1, p2 = points[line[0]], points[line[2]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2, alpha=0.7, zorder=1)
    else:  # The circular line
        # Draw as a circle passing through the three points
        center = (0, 0)
        circle_radius = radius * 0.9
        circle = Circle(center, circle_radius, fill=False, edgecolor='red', linewidth=2, 
                       alpha=0.7, zorder=1)
        ax.add_patch(circle)

# Add title
ax.text(0, 2.3, 'Fano Plane: The Muladhara 4', ha='center', va='center', 
        fontsize=16, fontweight='bold')

# Add subtitle
ax.text(0, 2.0, 'Anti-flags opposite to stabilizing point p₀', ha='center', va='center', 
        fontsize=12, style='italic')

# Create legend
blue_line = mlines.Line2D([], [], color='blue', linewidth=2, 
                         label='Lines through p₀ (3 lines)')
red_line = mlines.Line2D([], [], color='red', linewidth=2, 
                        label='Lines NOT through p₀ (4 lines)\n"The Muladhara 4"')
ax.legend(handles=[blue_line, red_line], loc='upper right', fontsize=11, 
          frameon=True, facecolor='white', edgecolor='gray')

# Add annotation about the anti-flag structure
ax.text(-2.2, -2.0, 'The 4 red lines form the unique orbit\nof anti-flags strictly opposite to p₀', 
        ha='left', va='top', fontsize=10, 
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gray", alpha=0.8))

plt.tight_layout()
plt.savefig('fano_muladhara_4.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()