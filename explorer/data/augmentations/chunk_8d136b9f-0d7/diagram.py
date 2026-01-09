import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Circle
from matplotlib.lines import Line2D

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.axis('off')

# Define the 7 points of the Fano plane
# Point p at center, 6 points in hexagon
p = np.array([0, 0])
radius = 1.5
angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 points
points = {
    'p': p,
    '1': radius * np.array([np.cos(angles[0]), np.sin(angles[0])]),
    '2': radius * np.array([np.cos(angles[1]), np.sin(angles[1])]),
    '3': radius * np.array([np.cos(angles[2]), np.sin(angles[2])]),
    '4': radius * np.array([np.cos(angles[3]), np.sin(angles[3])]),
    '5': radius * np.array([np.cos(angles[4]), np.sin(angles[4])]),
    '6': radius * np.array([np.cos(angles[5]), np.sin(angles[5])])
}

# Define the 7 lines of Fano plane (as sets of 3 points each)
lines = [
    ['p', '1', '2'],
    ['p', '3', '4'],
    ['p', '5', '6'],
    ['1', '3', '5'],
    ['2', '4', '6'],
    ['1', '4', '6'],
    ['2', '3', '5']
]

# Draw the lines
for line in lines:
    if 'p' in line:
        # Lines through p in lighter color
        for i in range(len(line)):
            for j in range(i+1, len(line)):
                p1 = points[line[i]]
                p2 = points[line[j]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', linewidth=1, alpha=0.5)
    else:
        # Circle connecting the three points (inscribed circle through them)
        pts = [points[pt] for pt in line]
        # For outer triangles, draw as curved lines
        center = np.mean(pts, axis=0)
        for i in range(3):
            p1 = pts[i]
            p2 = pts[(i+1)%3]
            # Create curved path
            t = np.linspace(0, 1, 50)
            curve = np.outer(1-t, p1) + np.outer(t, p2)
            # Add slight curve
            mid = 0.5 * (p1 + p2)
            offset = 0.1 * (mid - center) / np.linalg.norm(mid - center)
            curve[25] += offset
            ax.plot(curve[:, 0], curve[:, 1], 'gray', linewidth=1, alpha=0.5)

# Highlight triangles through p (12 triangles)
triangles_through_p = []
for i, line in enumerate(lines[:3]):  # First 3 lines contain p
    # Each line creates 4 triangles with the other 4 points
    other_points = [pt for pt in ['1', '2', '3', '4', '5', '6'] if pt not in line]
    line_points = [pt for pt in line if pt != 'p']
    for other in other_points:
        triangle = ['p'] + line_points[:1] + [other]
        triangles_through_p.append(triangle)
        # Draw these triangles
        pts = [points[pt] for pt in triangle]
        triangle_patch = Polygon(pts, fill=True, facecolor='lightblue', 
                               edgecolor='blue', alpha=0.3, linewidth=1.5)
        ax.add_patch(triangle_patch)

# Sample of triangles NOT containing p (showing anti-incidence)
sample_triangles_no_p = [
    ['1', '3', '5'],
    ['2', '4', '6'],
    ['1', '4', '6'],
    ['2', '3', '5']
]

for triangle in sample_triangles_no_p:
    pts = [points[pt] for pt in triangle]
    triangle_patch = Polygon(pts, fill=True, facecolor='lightcoral', 
                           edgecolor='red', alpha=0.2, linewidth=1, linestyle='--')
    ax.add_patch(triangle_patch)

# Draw points
for name, pt in points.items():
    if name == 'p':
        # Special styling for p as "Nada Brahman"
        circle = Circle(pt, 0.15, facecolor='gold', edgecolor='darkgoldenrod', 
                       linewidth=3, zorder=10)
        ax.add_patch(circle)
        ax.text(pt[0], pt[1]-0.4, 'p\n(Nada Brahman)', ha='center', va='top', 
                fontsize=10, fontweight='bold')
    else:
        circle = Circle(pt, 0.08, facecolor='black', zorder=10)
        ax.add_patch(circle)
        ax.text(pt[0]*1.15, pt[1]*1.15, name, ha='center', va='center', fontsize=10)

# Add arrows showing anti-incidence relation
# Sample arrow from p to a triangle not containing it
triangle_center = np.mean([points['1'], points['3'], points['5']], axis=0)
arrow_start = p + 0.2 * (triangle_center - p) / np.linalg.norm(triangle_center - p)
arrow_end = triangle_center - 0.3 * (triangle_center - p) / np.linalg.norm(triangle_center - p)
ax.annotate('', xy=arrow_end, xytext=arrow_start,
            arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.7))
ax.text((arrow_start[0] + arrow_end[0])/2 + 0.2, 
        (arrow_start[1] + arrow_end[1])/2, 
        'anti-incidence\n(x ∉ T)', ha='center', fontsize=8, style='italic')

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', 
           markeredgecolor='darkgoldenrod', markersize=10, label='p (Nada Brahman)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
           markersize=8, label='Other points'),
    plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.3, 
                  edgecolor='blue', label='12 triangles through p'),
    plt.Rectangle((0, 0), 1, 1, facecolor='lightcoral', alpha=0.2, 
                  edgecolor='red', linestyle='--', label='16 triangles not containing p'),
    Line2D([0], [0], color='red', linewidth=2, linestyle='-', 
           marker='>', markersize=8, label='Anti-incidence relation')
]

ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

# Title
ax.text(0, 2.8, 'Fano Plane: 112 Chakras as Point-Triangle Anti-Incidences', 
        ha='center', fontsize=14, fontweight='bold')
ax.text(0, 2.5, '7 points × 16 triangles = 112 anti-incidences {(x,T): x∉T}', 
        ha='center', fontsize=12, style='italic')

plt.tight_layout()
plt.show()