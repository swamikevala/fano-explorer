import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))

# Define colors
point_color = '#2E86AB'
line_color = '#A23B72'
antipoint_color = '#F18F01'
antiline_color = '#C73E1D'
projection_color = '#6A994E'

# Fano plane structure
fano_lines = [
    [0, 1, 2],  # Line 0
    [0, 3, 4],  # Line 1
    [0, 5, 6],  # Line 2
    [1, 3, 5],  # Line 3
    [1, 4, 6],  # Line 4
    [2, 3, 6],  # Line 5
    [2, 4, 5]   # Line 6
]

# Function to get points not on a line
def points_not_on_line(line_idx):
    on_line = set(fano_lines[line_idx])
    return [p for p in range(7) if p not in on_line]

# Layout 1: Fano Plane (top left)
ax1 = plt.subplot(2, 3, 1)
ax1.set_title('Fano Plane\n7 points, 7 lines', fontsize=12, fontweight='bold')
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_aspect('equal')
ax1.axis('off')

# Fano plane point positions
angles = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/2
fano_pos = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(6)}
fano_pos[6] = (0, 0)  # center point

# Draw Fano plane
for i, pos in fano_pos.items():
    circle = Circle(pos, 0.08, color=point_color, zorder=3)
    ax1.add_patch(circle)
    ax1.text(pos[0], pos[1], str(i), ha='center', va='center', 
             color='white', fontsize=8, fontweight='bold', zorder=4)

# Draw lines
line_coords = [
    [(fano_pos[0], fano_pos[1]), (fano_pos[1], fano_pos[2]), (fano_pos[2], fano_pos[0])],  # triangle
    [(fano_pos[0], fano_pos[3]), (fano_pos[3], fano_pos[4]), (fano_pos[4], fano_pos[0])],
    [(fano_pos[0], fano_pos[5]), (fano_pos[5], fano_pos[6]), (fano_pos[6], fano_pos[0])],
    [(fano_pos[1], fano_pos[3]), (fano_pos[3], fano_pos[5]), (fano_pos[5], fano_pos[1])],
    [(fano_pos[1], fano_pos[4]), (fano_pos[4], fano_pos[6]), (fano_pos[6], fano_pos[1])],
    [(fano_pos[2], fano_pos[3]), (fano_pos[3], fano_pos[6]), (fano_pos[6], fano_pos[2])],
    [(fano_pos[2], fano_pos[4]), (fano_pos[4], fano_pos[5]), (fano_pos[5], fano_pos[2])]
]

for i, line_path in enumerate(line_coords):
    for j in range(len(line_path)):
        start, end = line_path[j]
        ax1.plot([start[0], end[0]], [start[1], end[1]], 
                color=line_color, linewidth=2, alpha=0.7, zorder=1)

# Layout 2: Anti-flags visualization (top middle)
ax2 = plt.subplot(2, 3, 2)
ax2.set_title('28 Anti-flags\n(p, L) where p ∉ L', fontsize=12, fontweight='bold')
ax2.set_xlim(-0.5, 6.5)
ax2.set_ylim(-0.5, 6.5)
ax2.axis('off')

# Create grid showing anti-flags
for line_idx in range(7):
    for i, point in enumerate(points_not_on_line(line_idx)):
        x = point
        y = 6 - line_idx
        
        # Draw point not on line
        circle = Circle((x, y), 0.3, color=antipoint_color, alpha=0.8)
        ax2.add_patch(circle)
        ax2.text(x, y, f'{point}', ha='center', va='center', 
                color='white', fontsize=8, fontweight='bold')
        
        # Draw line indicator
        rect = FancyBboxPatch((x-0.35, y-0.45), 0.7, 0.15, 
                             boxstyle="round,pad=0.02", 
                             facecolor=antiline_color, alpha=0.6)
        ax2.add_patch(rect)
        ax2.text(x, y-0.37, f'L{line_idx}', ha='center', va='center', 
                fontsize=6, color='white')

# Add axis labels
ax2.text(-0.8, 3, 'Lines', rotation=90, ha='center', va='center', fontsize=10)
ax2.text(3, 7, 'Points not on line', ha='center', fontsize=10)

# Layout 3: Pointed anti-flags sample (top right)
ax3 = plt.subplot(2, 3, 3)
ax3.set_title('84 Pointed Anti-flags\n(p, L, r) where p ∉ L, r ∈ L', 
              fontsize=12, fontweight='bold')
ax3.set_xlim(-0.5, 3.5)
ax3.set_ylim(-0.5, 4)
ax3.axis('off')

# Show example: anti-flag (0, L5) with its 3 pointed versions
example_line = 5
example_point = 0
points_on_L5 = fano_lines[example_line]

ax3.text(1.5, 3.5, f'Example: p={example_point}, L={example_line}', 
         ha='center', fontsize=10, fontweight='bold')

# Draw the anti-flag
circle = Circle((0.5, 2.5), 0.3, color=antipoint_color)
ax3.add_patch(circle)
ax3.text(0.5, 2.5, '0', ha='center', va='center', 
         color='white', fontsize=10, fontweight='bold')

rect = FancyBboxPatch((0.15, 2.05), 0.7, 0.15, 
                     boxstyle="round,pad=0.02", 
                     facecolor=antiline_color)
ax3.add_patch(rect)
ax3.text(0.5, 2.12, 'L5', ha='center', va='center', 
         fontsize=8, color='white')

# Draw arrows to pointed versions
for i, r in enumerate(points_on_L5):
    x = 2.5
    y = 3 - i * 1
    
    # Arrow
    arrow = FancyArrowPatch((1, 2.5), (x-0.5, y),
                           arrowstyle='->', mutation_scale=15,
                           color='gray', alpha=0.6)
    ax3.add_patch(arrow)
    
    # Pointed anti-flag representation
    circle = Circle((x, y), 0.25, color=antipoint_color)
    ax3.add_patch(circle)
    ax3.text(x, y, '0', ha='center', va='center', 
             color='white', fontsize=8, fontweight='bold')
    
    rect = FancyBboxPatch((x-0.3, y-0.4), 0.6, 0.12, 
                         boxstyle="round,pad=0.02", 
                         facecolor=antiline_color)
    ax3.add_patch(rect)
    ax3.text(x, y-0.34, 'L5', ha='center', va='center', 
             fontsize=6, color='white')
    
    # Highlight the chosen point
    circle_r = Circle((x+0.6, y), 0.15, color=point_color)
    ax3.add_patch(circle_r)
    ax3.text(x+0.6, y, str(r), ha='center', va='center', 
             color='white', fontsize=8, fontweight='bold')
    ax3.text(x+0.6, y-0.3, 'r ∈ L5', ha='center', fontsize=6)

# Layout 4: Projection structure (bottom, spanning all columns)
ax4 = plt.subplot(2, 1, 2)
ax4.set_title('Projection π: Pointed Anti-flags → Anti-flags (3-to-1)', 
              fontsize=14, fontweight='bold')
ax4.set_xlim(-1, 11)
ax4.set_ylim(-0.5, 3.5)
ax4.axis('off')

# Show fiber structure
fiber_examples = [(1, 3), (4, 2), (6, 0)]  # (point, line) pairs

for idx, (p, L) in enumerate(fiber_examples):
    x_base = idx * 4 + 1
    
    # Draw base anti-flag
    base_y = 0.5
    circle = Circle((x_base, base_y), 0.3, color=antipoint_color)
    ax4.add_patch(circle)
    ax4.text(x_base, base_y, str(p), ha='center', va='center', 
             color='white', fontsize=10, fontweight='bold')
    
    rect = FancyBboxPatch((x_base-0.35, base_y-0.45), 0.7, 0.15, 
                         boxstyle="round,pad=0.02", 
                         facecolor=antiline_color)
    ax4.add_patch(rect)
    ax4.text(x_base, base_y-0.37, f'L{L}', ha='center', va='center', 
             fontsize=8, color='white')
    
    # Draw fiber (3 pointed anti-flags)
    points_on_line = fano_lines[L]
    for i, r in enumerate(points_on_line):
        x = x_base + (i-1) * 0.8
        y = 2.5
        
        # Pointed anti-flag
        circle = Circle((x, y), 0.25, color=antipoint_color, alpha=0.8)
        ax4.add_patch(circle)
        ax4.text(x, y, str(p), ha='center', va='center', 
                 color='white', fontsize=8, fontweight='bold')
        
        # Highlight r
        circle_r = Circle((x+0.35, y), 0.12, color=point_color)
        ax4.add_patch(circle_r)
        ax4.text(x+0.35, y, str(r), ha='center', va='center', 
                 color='white', fontsize=7, fontweight='bold')
        
        # Projection arrow
        arrow = FancyArrowPatch((x, y-0.3), (x_base, base_y+0.35),
                               arrowstyle='->', mutation_scale=12,
                               color=projection_color, linewidth=2, alpha=0.7)
        ax4.add_patch(arrow)

# Add labels
ax4.text(-0.5, 2.5, '84 Pointed\nAnti-flags', ha='center', va='center', 
         fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
         facecolor='lightgray', alpha=0.5))
ax4.text(-0.5, 0.5, '28 Anti-flags', ha='center', va='center', 
         fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
         facecolor='lightgray', alpha=0.5))

ax4.text(10.5, 1.5, 'π', ha='center', va='center', 
         fontsize=16, fontweight='bold', color=projection_color)

# Add legend
legend_elements = [
    patches.Patch(color=point_color, label='Points in L'),
    patches.Patch(color=antipoint_color, label='Points not in L'),
    patches.Patch(color=line_color, label='Fano lines'),
    patches.Patch(color=antiline_color, label='Line indicators'),
    patches.Patch(color=projection_color, label='Projection π')
]

ax4.legend(handles=legend_elements, loc='lower right', 
          bbox_to_anchor=(1.1, -0.1), ncol=5, frameon=True, 
          fancybox=True, fontsize=9)

plt.tight_layout()
plt.savefig('chakra_fano_projection.png', dpi=300, bbox_inches='tight')
plt.show()