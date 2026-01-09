import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Circle
import matplotlib.patches as mpatches

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Fano plane with anti-incidence highlighted
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-2.5, 2.5)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('Anti-incidence in Fano Plane', fontsize=14, pad=20)

# Define Fano plane points
angles = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/2
radius = 1.5
points = [(radius * np.cos(a), radius * np.sin(a)) for a in angles[:-1]]
points.append((0, 0))  # Center point

# Label points
labels = ['1', '2', '3', '4', '5', '6', '7']
for i, (x, y) in enumerate(points):
    ax1.plot(x, y, 'ko', markersize=8)
    offset = 0.25 if i < 6 else 0
    ax1.text(x*(1+offset/radius), y*(1+offset/radius), labels[i], 
             ha='center', va='center', fontsize=11, fontweight='bold')

# Draw all lines (edges)
lines = [
    (0, 1, 3), (1, 2, 4), (2, 0, 5),  # Sides of outer triangle
    (0, 4, 6), (1, 5, 6), (2, 3, 6),  # Lines through center
    (3, 4, 5)  # Inscribed circle
]

for line in lines:
    if 6 in line:  # Lines through center
        for i in line[:2]:
            ax1.plot([points[i][0], points[6][0]], [points[i][1], points[6][1]], 
                    'k-', linewidth=1, alpha=0.3)
    else:  # Other lines
        for i in range(3):
            p1, p2 = line[i], line[(i+1)%3]
            ax1.plot([points[p1][0], points[p2][0]], [points[p1][1], points[p2][1]], 
                    'k-', linewidth=1, alpha=0.3)

# Highlight specific anti-incidence: point x=1 and triangle T=(2,4,5)
x_point = 0  # Point 1
triangle = [1, 3, 4]  # Points 2, 4, 5

# Draw the triangle T in blue
triangle_points = [points[i] for i in triangle]
triangle_patch = Polygon(triangle_points, fill=False, edgecolor='blue', 
                        linewidth=3, linestyle='-')
ax1.add_patch(triangle_patch)

# Highlight point x in red
ax1.plot(points[x_point][0], points[x_point][1], 'ro', markersize=12)

# Add labels
ax1.text(points[x_point][0]-0.3, points[x_point][1]+0.3, 'x (seer)', 
         color='red', fontsize=10, fontweight='bold')
ax1.text(0, -0.7, 'T (field of three)', color='blue', fontsize=10, 
         ha='center', fontweight='bold')
ax1.text(0, -2.2, 'x ∉ T', fontsize=12, ha='center', style='italic')

# Subplot 2: Partition diagram
ax2.set_xlim(-1, 8)
ax2.set_ylim(-1, 5)
ax2.axis('off')
ax2.set_title('112 Anti-incidences Partitioned by Seer', fontsize=14, pad=20)

# Draw 7 vertical groups
group_width = 0.8
group_spacing = 1.0
y_base = 1.5

for i in range(7):
    x_center = i * group_spacing + 1
    
    # Draw point (seer) at top
    ax2.plot(x_center, y_base + 2, 'ko', markersize=10)
    ax2.text(x_center, y_base + 2.3, str(i+1), ha='center', va='bottom', 
             fontsize=11, fontweight='bold')
    
    # Draw box representing 16 anti-incidences
    rect = plt.Rectangle((x_center - group_width/2, y_base - 1), 
                        group_width, 1.5, 
                        fill=True, facecolor='lightblue', 
                        edgecolor='blue', linewidth=2)
    ax2.add_patch(rect)
    
    # Add count
    ax2.text(x_center, y_base - 0.25, '16', ha='center', va='center', 
             fontsize=10, fontweight='bold')
    
    # Draw arrow from point to box
    ax2.arrow(x_center, y_base + 1.8, 0, -1.2, 
              head_width=0.1, head_length=0.1, 
              fc='gray', ec='gray', alpha=0.7)

# Add total count
ax2.text(3.5, 0.2, '7 × 16 = 112 anti-incidences', 
         ha='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

# Add legend
ax2.text(3.5, 4.5, 'Each point sees 16 triangles it doesn\'t belong to',
         ha='center', fontsize=10, style='italic')

# Create legend for first subplot
red_patch = mpatches.Patch(color='red', label='Witness point x')
blue_patch = mpatches.Patch(color='blue', label='Field triangle T')
ax1.legend(handles=[red_patch, blue_patch], loc='lower left', fontsize=9)

plt.tight_layout()
plt.savefig('fano_antiincidences.png', dpi=300, bbox_inches='tight')
plt.show()