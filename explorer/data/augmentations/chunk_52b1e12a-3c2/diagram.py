import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon
import matplotlib.patches as mpatches

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axis('off')

# Define 7 points of Fano plane on unit circle (plus center)
angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
points = [(np.cos(a), np.sin(a)) for a in angles]
points.append((0, 0))  # Add center point (index 7)

# Define the 7 lines of Fano plane
# Each line contains 3 points (given by indices)
lines = [
    [0, 1, 3],  # Line 1
    [1, 2, 4],  # Line 2
    [2, 3, 5],  # Line 3
    [3, 4, 6],  # Line 4
    [4, 5, 0],  # Line 5
    [5, 6, 1],  # Line 6
    [6, 0, 2],  # Line 7
]

# Choose point p (let's choose point 0)
p_index = 0
p_coord = points[p_index]

# Find lines incident to p
incident_lines = []
non_incident_lines = []
for i, line in enumerate(lines):
    if p_index in line:
        incident_lines.append((i, line))
    else:
        non_incident_lines.append((i, line))

# Draw all points
for i, (x, y) in enumerate(points[:7]):
    if i == p_index:
        ax.plot(x, y, 'ro', markersize=15, markeredgewidth=2, 
                markeredgecolor='darkred', label='Point p')
    else:
        ax.plot(x, y, 'ko', markersize=10)
    ax.text(x*1.15, y*1.15, f'{i}', ha='center', va='center', fontsize=12)

# Draw center point
ax.plot(0, 0, 'ko', markersize=10)
ax.text(0.1, -0.1, '7', ha='center', va='center', fontsize=12)

# Draw incident lines (3 lines through p) in red
for i, line in incident_lines:
    if 7 in line:  # Line through center
        idx1, idx2 = [j for j in line if j != 7]
        ax.plot([points[idx1][0], points[idx2][0]], 
                [points[idx1][1], points[idx2][1]], 'r-', linewidth=2, alpha=0.7)
    else:  # Circle arc
        # Draw as straight lines for simplicity
        for j in range(len(line)):
            idx1, idx2 = line[j], line[(j+1)%3]
            ax.plot([points[idx1][0], points[idx2][0]], 
                    [points[idx1][1], points[idx2][1]], 'r-', linewidth=2, alpha=0.7)

# Draw non-incident lines (4 lines not through p) in blue
for i, line in non_incident_lines:
    if 7 in line:  # Line through center
        idx1, idx2 = [j for j in line if j != 7]
        ax.plot([points[idx1][0], points[idx2][0]], 
                [points[idx1][1], points[idx2][1]], 'b-', linewidth=1.5, alpha=0.5)
    else:  # Circle arc
        for j in range(len(line)):
            idx1, idx2 = line[j], line[(j+1)%3]
            ax.plot([points[idx1][0], points[idx2][0]], 
                    [points[idx1][1], points[idx2][1]], 'b-', linewidth=1.5, alpha=0.5)

# Highlight some tangential triangles formed by non-incident lines
# Show 3 example triangles (out of 12 possible)
triangle_examples = [
    [points[1], points[2], points[4]],  # Triangle from line [1,2,4]
    [points[3], points[4], points[6]],  # Triangle from line [3,4,6]
    [points[5], points[6], points[1]],  # Triangle from line [5,6,1]
]

for i, triangle in enumerate(triangle_examples):
    poly = Polygon(triangle, fill=True, facecolor='lightblue', 
                   edgecolor='blue', alpha=0.3, linewidth=1)
    ax.add_patch(poly)

# Add title and annotations
ax.set_title('Fano Plane: 7×12 = 84 Structure', fontsize=16, pad=20)

# Add text annotations
ax.text(0, -1.35, 'With p chosen:', ha='center', fontsize=12, weight='bold')
ax.text(0, -1.45, '• 3 lines incident to p (red)', ha='center', fontsize=11)
ax.text(0, -1.55, '• 4 lines non-incident to p (blue)', ha='center', fontsize=11)
ax.text(0, -1.65, '• 4 lines × 3 points each = 12 tangential triangles', ha='center', fontsize=11)
ax.text(0, -1.75, '• 7 choices of p × 12 triangles = 84 total', ha='center', fontsize=11, weight='bold')

# Add legend
red_line = mpatches.Patch(color='red', label='Lines through p')
blue_line = mpatches.Patch(color='blue', label='Lines not through p')
blue_triangle = mpatches.Patch(color='lightblue', label='Example triangles (3 of 12)')
ax.legend(handles=[red_line, blue_line, blue_triangle], loc='upper right')

plt.tight_layout()
plt.show()