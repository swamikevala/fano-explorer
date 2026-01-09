import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon
import matplotlib.lines as mlines

# Set up the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Define Fano plane points in symmetric arrangement
angle_offset = -np.pi/2  # Start from top
angles = np.linspace(0, 2*np.pi, 7, endpoint=False) + angle_offset
radius = 2

# Outer 6 points on hexagon
hex_points = [(radius * np.cos(angles[i]), radius * np.sin(angles[i])) for i in range(6)]
# Center point
center_point = (0, 0)
# All 7 points
all_points = hex_points + [center_point]

# Define the 7 lines of the Fano plane
# Each line contains exactly 3 points (indices)
fano_lines = [
    [0, 1, 3],  # Line 1
    [1, 2, 4],  # Line 2
    [2, 3, 5],  # Line 3
    [3, 4, 0],  # Line 4
    [4, 5, 1],  # Line 5
    [5, 0, 2],  # Line 6
    [0, 6, 4],  # Line 7 (through center)
    [1, 6, 5],  # Line 8 (through center)
    [2, 6, 3]   # Line 9 (through center)
]

# We'll show two different triangles as examples
# Triangle 1: points 0, 1, 6 (includes center)
# Triangle 2: points 0, 2, 4 (no center)

def draw_fano_plane(ax, triangle_indices, title):
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Draw all Fano lines in light gray
    for line in fano_lines:
        if 6 in line:  # Lines through center
            for i in range(2):
                if line[i] != 6:
                    p1 = all_points[line[i]]
                    p2 = all_points[6]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'lightgray', linewidth=1, zorder=1)
        else:  # Circular arc for outer lines
            p1, p2, p3 = [all_points[i] for i in line]
            # Just draw as straight lines for simplicity
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'lightgray', linewidth=1, zorder=1)
            ax.plot([p2[0], p3[0]], [p2[1], p3[1]], 'lightgray', linewidth=1, zorder=1)
            ax.plot([p3[0], p1[0]], [p3[1], p1[1]], 'lightgray', linewidth=1, zorder=1)
    
    # Highlight the triangle T
    triangle_points = [all_points[i] for i in triangle_indices]
    triangle = Polygon(triangle_points, fill=False, edgecolor='red', linewidth=3, zorder=2)
    ax.add_patch(triangle)
    
    # Find which lines contain edges of the triangle
    edge_lines = []
    for line in fano_lines:
        # Check each edge of the triangle
        edges = [(triangle_indices[0], triangle_indices[1]),
                 (triangle_indices[1], triangle_indices[2]),
                 (triangle_indices[2], triangle_indices[0])]
        for edge in edges:
            if edge[0] in line and edge[1] in line:
                edge_lines.append(line)
    
    # Draw edge lines in blue
    for line in edge_lines:
        if 6 in line:  # Lines through center
            for i in range(2):
                if line[i] != 6:
                    p1 = all_points[line[i]]
                    p2 = all_points[6]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'blue', linewidth=2, zorder=3)
        else:  # Outer lines
            p1, p2, p3 = [all_points[i] for i in line]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'blue', linewidth=2, zorder=3)
            ax.plot([p2[0], p3[0]], [p2[1], p3[1]], 'blue', linewidth=2, zorder=3)
            ax.plot([p3[0], p1[0]], [p3[1], p1[1]], 'blue', linewidth=2, zorder=3)
    
    # Identify conductive and radical points
    external_points = [i for i in range(7) if i not in triangle_indices]
    conductive_points = []
    radical_points = []
    
    for point_idx in external_points:
        on_edge_line = False
        for line in edge_lines:
            if point_idx in line:
                on_edge_line = True
                break
        if on_edge_line:
            conductive_points.append(point_idx)
        else:
            radical_points.append(point_idx)
    
    # Draw all points
    for i, point in enumerate(all_points):
        if i in triangle_indices:
            # Triangle vertices
            ax.plot(point[0], point[1], 'o', color='red', markersize=12, zorder=5)
            ax.text(point[0], point[1], str(i), ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white', zorder=6)
        elif i in conductive_points:
            # Conductive points (on edge lines)
            ax.plot(point[0], point[1], 'o', color='green', markersize=12, zorder=5)
            ax.text(point[0], point[1], str(i), ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white', zorder=6)
        else:
            # Radical points (not on edge lines)
            ax.plot(point[0], point[1], 'o', color='orange', markersize=12, zorder=5)
            ax.text(point[0], point[1], str(i), ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white', zorder=6)
    
    # Add count labels
    ax.text(0, -2.8, f"Conductive points: {len(conductive_points)}", 
           ha='center', fontsize=12, color='green', fontweight='bold')
    ax.text(0, -3.2, f"Radical points: {len(radical_points)}", 
           ha='center', fontsize=12, color='orange', fontweight='bold')

# Draw two examples
draw_fano_plane(ax1, [0, 1, 6], "Example 1: Triangle {0, 1, 6}")
draw_fano_plane(ax2, [0, 2, 4], "Example 2: Triangle {0, 2, 4}")

# Add legend
legend_elements = [
    mlines.Line2D([0], [0], color='red', lw=3, label='Triangle T'),
    mlines.Line2D([0], [0], color='blue', lw=2, label='Edge lines of T'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                  markersize=10, label='Conductive points (on edge lines)'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                  markersize=10, label='Radical points (off all edge lines)')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=12, 
          frameon=True, bbox_to_anchor=(0.5, -0.05))

# Add main title
fig.suptitle('Anti-flags in the Fano Plane: Conductive vs Radical States', 
            fontsize=16, fontweight='bold')

# Add explanation text
fig.text(0.5, 0.02, 'For any triangle T in the Fano plane, the 4 external points split into 3 conductive and 1 radical',
         ha='center', fontsize=11, style='italic')

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig('fano_antiflags.png', dpi=300, bbox_inches='tight')
plt.show()