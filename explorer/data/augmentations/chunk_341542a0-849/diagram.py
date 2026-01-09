import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Circle
import matplotlib.patches as mpatches

# Define the 7 points of the Fano plane in a symmetric arrangement
angles = np.linspace(0, 2*np.pi, 7, endpoint=False) - np.pi/2
radius = 2
points = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
# Center point
points.append((0, 0))

# Define point labels
labels = ['a', 'b', 'c', 'a+b', 'b+c', 'c+a', 'a+b+c', '0']

# Define the 7 lines of the Fano plane (as indices)
lines = [
    [0, 1, 3],      # a, b, a+b
    [1, 2, 4],      # b, c, b+c
    [2, 0, 5],      # c, a, c+a
    [0, 4, 6],      # a, b+c, a+b+c
    [1, 5, 6],      # b, c+a, a+b+c
    [2, 3, 6],      # c, a+b, a+b+c
    [3, 4, 5]       # a+b, b+c, c+a
]

# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Fano Triangle Complements: Canonical Decomposition into Opposite Line + Sum Point', 
             fontsize=14, y=0.98)

# Define three different triangles to show
triangles = [
    {'vertices': [0, 1, 2], 'name': 'T₁ = {a, b, c}'},
    {'vertices': [0, 1, 4], 'name': 'T₂ = {a, b, b+c}'},
    {'vertices': [3, 4, 5], 'name': 'T₃ = {a+b, b+c, c+a}'}
]

for idx, (ax, triangle_data) in enumerate(zip(axes, triangles)):
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw all lines first (light gray)
    for line in lines:
        if len(line) == 3:
            # Regular line
            for i in range(3):
                p1, p2 = points[line[i]], points[line[(i+1)%3]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'lightgray', linewidth=1, zorder=1)
        
    # Draw the circle line
    circle_line = [3, 4, 5]
    t = np.linspace(0, 2*np.pi, 100)
    circle_radius = 1.2
    ax.plot(circle_radius * np.cos(t), circle_radius * np.sin(t), 'lightgray', linewidth=1, zorder=1)
    
    # Get triangle vertices
    triangle_vertices = triangle_data['vertices']
    
    # Draw and fill the triangle
    triangle_points = [points[i] for i in triangle_vertices]
    triangle_patch = Polygon(triangle_points, alpha=0.3, facecolor='blue', edgecolor='blue', linewidth=2)
    ax.add_patch(triangle_patch)
    
    # Find complement points
    all_points = set(range(7))  # Excluding center point
    triangle_set = set(triangle_vertices)
    complement = list(all_points - triangle_set)
    
    # Find the opposite line (3 collinear points in complement)
    opposite_line = None
    sum_point = None
    
    for line in lines:
        line_set = set(line)
        if len(line_set.intersection(triangle_set)) == 0 and line_set.issubset(complement):
            opposite_line = line
            sum_point = list(set(complement) - set(line))[0]
            break
    
    # Highlight opposite line
    if opposite_line:
        # Draw the opposite line in red
        if set(opposite_line) == {3, 4, 5}:  # Circle line
            t = np.linspace(0, 2*np.pi, 100)
            ax.plot(circle_radius * np.cos(t), circle_radius * np.sin(t), 'red', linewidth=3, zorder=2)
        else:
            for i in range(3):
                p1, p2 = points[opposite_line[i]], points[opposite_line[(i+1)%3]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'red', linewidth=3, zorder=2)
    
    # Draw all points
    for i, (x, y) in enumerate(points[:-1]):  # Exclude center
        if i in triangle_vertices:
            ax.plot(x, y, 'o', markersize=12, color='blue', markeredgecolor='darkblue', 
                   markeredgewidth=2, zorder=4)
        elif i in opposite_line:
            ax.plot(x, y, 'o', markersize=12, color='red', markeredgecolor='darkred', 
                   markeredgewidth=2, zorder=4)
        elif i == sum_point:
            ax.plot(x, y, 's', markersize=14, color='green', markeredgecolor='darkgreen', 
                   markeredgewidth=2, zorder=4)
        else:
            ax.plot(x, y, 'o', markersize=8, color='lightgray', zorder=3)
    
    # Add labels
    for i, (x, y) in enumerate(points[:-1]):
        if i < len(labels) - 1:
            # Adjust label positions to avoid overlap
            offset = 0.35
            angle = angles[i] if i < 7 else 0
            dx = offset * np.cos(angle) if i < 7 else 0
            dy = offset * np.sin(angle) if i < 7 else 0.3
            ax.text(x + dx, y + dy, labels[i], ha='center', va='center', fontsize=11, 
                   fontweight='bold' if i in triangle_vertices or i == sum_point else 'normal')
    
    # Add title and annotations
    ax.set_title(triangle_data['name'], fontsize=12, pad=10)
    
    # Add complement decomposition text
    if opposite_line and sum_point is not None:
        opp_labels = [labels[i] for i in sorted(opposite_line)]
        complement_text = f"Tᶜ = {{{', '.join(opp_labels)}}} ∪ {{{labels[sum_point]}}}"
        ax.text(0, -2.7, complement_text, ha='center', va='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

# Add legend
legend_elements = [
    mpatches.Patch(color='blue', alpha=0.3, label='Triangle T'),
    plt.Line2D([0], [0], color='red', linewidth=3, label='Opposite line L(T)'),
    plt.Line2D([0], [0], marker='s', color='green', linewidth=0, markersize=10, 
               label='Sum point s(T)')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, frameon=True, 
          bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.show()