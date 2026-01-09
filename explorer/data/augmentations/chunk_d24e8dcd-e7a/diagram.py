import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon
import matplotlib.patches as mpatches

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Define the 7 points of the Fano plane in a symmetric arrangement
# Points arranged as: 1 center, 6 on a hexagon
angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 points on circle
radius = 2

# Point positions
points = {
    0: np.array([0, 0]),  # center
    1: np.array([radius * np.cos(angles[0]), radius * np.sin(angles[0])]),
    2: np.array([radius * np.cos(angles[1]), radius * np.sin(angles[1])]),
    3: np.array([radius * np.cos(angles[2]), radius * np.sin(angles[2])]),
    4: np.array([radius * np.cos(angles[3]), radius * np.sin(angles[3])]),
    5: np.array([radius * np.cos(angles[4]), radius * np.sin(angles[4])]),
    6: np.array([radius * np.cos(angles[5]), radius * np.sin(angles[5])])
}

# Define the 7 lines (triangles) of the Fano plane
lines = [
    (0, 1, 2),
    (0, 3, 4),
    (0, 5, 6),
    (1, 3, 5),
    (2, 4, 6),
    (1, 4, 6),
    (2, 3, 5)
]

# Draw all lines as light gray
for line in lines:
    for i in range(3):
        p1 = points[line[i]]
        p2 = points[line[(i+1)%3]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'lightgray', linewidth=1.5, zorder=1)

# Highlight observer point (point 0) and one non-incident triangle (2, 4, 6)
observer = 0
field_triangle = (2, 4, 6)

# Draw the field triangle in blue
triangle_points = np.array([points[field_triangle[0]], 
                           points[field_triangle[1]], 
                           points[field_triangle[2]]])
field_poly = Polygon(triangle_points, fill=False, edgecolor='blue', 
                     linewidth=3, linestyle='-', zorder=3)
ax.add_patch(field_poly)

# Draw connecting lines of the field triangle
for i in range(3):
    p1 = points[field_triangle[i]]
    p2 = points[field_triangle[(i+1)%3]]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'blue', linewidth=3, zorder=3)

# Draw all points
for i, pos in points.items():
    if i == observer:
        # Observer point in red
        circle = Circle(pos, 0.15, facecolor='red', edgecolor='darkred', 
                       linewidth=2, zorder=5)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1]-0.4, 'Observer (x)', fontsize=12, 
                ha='center', va='top', weight='bold', color='darkred')
    elif i in field_triangle:
        # Field points in blue
        circle = Circle(pos, 0.15, facecolor='lightblue', edgecolor='blue', 
                       linewidth=2, zorder=5)
        ax.add_patch(circle)
    else:
        # Other points in gray
        circle = Circle(pos, 0.1, facecolor='lightgray', edgecolor='gray', 
                       linewidth=1.5, zorder=4)
        ax.add_patch(circle)
    
    # Add point labels
    ax.text(pos[0], pos[1], str(i), fontsize=10, ha='center', va='center', 
            weight='bold', zorder=6)

# Add field label
field_center = np.mean(triangle_points, axis=0)
ax.text(field_center[0], field_center[1], 'Field (T)', fontsize=12, 
        ha='center', va='center', weight='bold', color='blue', 
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Add anti-incidence notation
ax.text(0, 3.2, 'Anti-incidence: x ∉ T', fontsize=14, ha='center', 
        weight='bold', style='italic')
ax.text(0, 2.8, 'Point 0 is not on triangle {2,4,6}', fontsize=11, 
        ha='center', color='gray')

# Count display
ax.text(-3.5, 2.5, 'Each point has\n16 non-incident\ntriangles', 
        fontsize=11, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))

# Total count
ax.text(3.5, 2.5, '7 points × 16 =\n112 observer-field\npairs', 
        fontsize=11, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

# Create legend
observer_patch = mpatches.Patch(color='red', label='Observer point')
field_patch = mpatches.Patch(color='blue', label='Field triangle')
other_patch = mpatches.Patch(color='lightgray', label='Other elements')
ax.legend(handles=[observer_patch, field_patch, other_patch], 
          loc='lower center', ncol=3, framealpha=0.9)

# Set axis properties
ax.set_xlim(-4, 4)
ax.set_ylim(-3.5, 3.5)
ax.set_aspect('equal')
ax.axis('off')

# Title
ax.set_title('Fano Plane: Observer-Field Anti-incidence Structure', 
             fontsize=16, weight='bold', pad=20)

plt.tight_layout()
plt.show()