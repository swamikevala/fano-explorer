import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(-1, 29)
ax.set_ylim(-1, 8)
ax.axis('off')

# Colors
base_color = '#2E4057'  # Dark blue-gray for stationary
flow_color = '#048A81'  # Teal for flowing
arrow_color = '#54C6EB'  # Light blue for projections
text_color = '#1A1A1A'

# Title
ax.text(14, 7.5, 'Fiber Bundle Structure: T → A', 
        fontsize=16, weight='bold', ha='center', color=text_color)

# Draw base level (28 anti-flags)
base_y = 1
for i in range(28):
    x_pos = i
    # Draw base node
    rect = FancyBboxPatch((x_pos - 0.3, base_y - 0.2), 0.6, 0.4,
                          boxstyle="round,pad=0.05",
                          facecolor=base_color, edgecolor='none')
    ax.add_patch(rect)
    # Label every 7th node
    if i % 7 == 0 or i == 27:
        ax.text(x_pos, base_y - 0.6, f'a_{i+1}', 
                fontsize=9, ha='center', color=text_color)

# Draw fiber level (84 pointed anti-flags)
fiber_y = 5
for i in range(28):
    x_base = i
    # Draw 3 fiber nodes for each base
    for j in range(3):
        x_offset = (j - 1) * 0.25
        x_fiber = x_base + x_offset
        # Draw fiber node
        circle = Circle((x_fiber, fiber_y), 0.12,
                       facecolor=flow_color, edgecolor='none')
        ax.add_patch(circle)
        
        # Draw projection arrow
        arrow = FancyArrowPatch((x_fiber, fiber_y - 0.12), 
                               (x_base, base_y + 0.25),
                               arrowstyle='->', 
                               connectionstyle="arc3,rad=0",
                               color=arrow_color, 
                               linewidth=0.5, 
                               alpha=0.6)
        ax.add_patch(arrow)

# Add level labels
ax.text(-0.5, fiber_y, '84 Pointed\nAnti-Flags\n(Flowing)', 
        fontsize=11, ha='right', va='center', color=text_color)
ax.text(-0.5, base_y, '28 Anti-Flags\n(Stationary)', 
        fontsize=11, ha='right', va='center', color=text_color)

# Add projection symbol
ax.text(14, 3, 'π', fontsize=20, ha='center', va='center', 
        style='italic', color=arrow_color)

# Add count annotations
ax.text(26, fiber_y + 0.8, '3 fibers per base', 
        fontsize=10, ha='center', color=flow_color,
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=flow_color))
ax.text(14, base_y - 1.2, '28 base elements', 
        fontsize=10, ha='center', color=base_color,
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=base_color))

# Add total count
ax.text(14, 6.5, '112 Total Elements = 84 + 28', 
        fontsize=12, ha='center', color=text_color, style='italic')

# Legend
legend_x = 1
legend_y = -0.5
ax.text(legend_x - 0.5, legend_y, 'Legend:', fontsize=10, weight='bold', color=text_color)
circle_legend = Circle((legend_x + 0.5, legend_y), 0.1, facecolor=flow_color, edgecolor='none')
ax.add_patch(circle_legend)
ax.text(legend_x + 0.8, legend_y, 'Flowing (Dynamic)', fontsize=9, va='center', color=text_color)
rect_legend = FancyBboxPatch((legend_x + 3.7, legend_y - 0.1), 0.4, 0.2,
                            boxstyle="round,pad=0.03",
                            facecolor=base_color, edgecolor='none')
ax.add_patch(rect_legend)
ax.text(legend_x + 4.3, legend_y, 'Stationary (Static)', fontsize=9, va='center', color=text_color)

plt.tight_layout()
plt.savefig('fiber_bundle_structure.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()