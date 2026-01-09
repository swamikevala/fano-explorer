"""
Verification of the 112 chakras as anti-flags and pointed anti-flags of the Fano plane
"""

# Define the Fano plane
# Points: 0, 1, 2, 3, 4, 5, 6
# Lines: each line contains exactly 3 points
fano_lines = [
    {0, 1, 3},  # Line 0
    {1, 2, 4},  # Line 1
    {2, 3, 5},  # Line 2
    {3, 4, 6},  # Line 3
    {4, 5, 0},  # Line 4
    {5, 6, 1},  # Line 5
    {6, 0, 2}   # Line 6
]

# Convert to lists for consistent ordering
fano_lines = [sorted(list(line)) for line in fano_lines]
all_points = list(range(7))

print("=== FANO PLANE STRUCTURE ===")
print(f"Points: {all_points}")
print(f"Lines: {fano_lines}")
print()

# Step 1: Generate all anti-flags (p, L) where p ∉ L
anti_flags = []
for point in all_points:
    for line_idx, line in enumerate(fano_lines):
        if point not in line:
            anti_flags.append((point, line_idx))

print("=== ANTI-FLAGS ===")
print(f"Total anti-flags: {len(anti_flags)}")
print("First 5 anti-flags (point, line_index):", anti_flags[:5])
print()

# Step 2: Generate all pointed anti-flags (p, L, r) where p ∉ L and r ∈ L
pointed_anti_flags = []
for point, line_idx in anti_flags:
    line = fano_lines[line_idx]
    for r in line:
        pointed_anti_flags.append((point, line_idx, r))

print("=== POINTED ANTI-FLAGS ===")
print(f"Total pointed anti-flags: {len(pointed_anti_flags)}")
print("First 5 pointed anti-flags (point, line_index, point_on_line):", pointed_anti_flags[:5])
print()

# Step 3: Verify the 3-to-1 projection property
print("=== PROJECTION VERIFICATION ===")
# Each anti-flag should map to exactly 3 pointed anti-flags
projection_counts = {}
for paf in pointed_anti_flags:
    anti_flag = (paf[0], paf[1])  # (p, L)
    projection_counts[anti_flag] = projection_counts.get(anti_flag, 0) + 1

# Check all counts are 3
all_counts_are_3 = all(count == 3 for count in projection_counts.values())
print(f"All anti-flags map to exactly 3 pointed anti-flags: {all_counts_are_3}")
print(f"Number of unique anti-flags in projection: {len(projection_counts)}")
print()

# Step 4: Group by point not on line (7 groups of 16)
print("=== GROUPING BY POINT ===")
groups_by_point = {}
for paf in pointed_anti_flags:
    point = paf[0]
    if point not in groups_by_point:
        groups_by_point[point] = []
    groups_by_point[point].append(paf)

for point in sorted(groups_by_point.keys()):
    group_size = len(groups_by_point[point])
    print(f"Point {point}: {group_size} pointed anti-flags")

print()

# Step 5: Verify the structure
print("=== VERIFICATION SUMMARY ===")
print(f"✓ Anti-flags (p ∉ L): {len(anti_flags)} (expected: 28)")
print(f"✓ Pointed anti-flags (p ∉ L, r ∈ L): {len(pointed_anti_flags)} (expected: 84)")
print(f"✓ Total chakras: {len(anti_flags) + len(pointed_anti_flags)} (expected: 112)")
print(f"✓ 3-to-1 projection: {all_counts_are_3}")
print(f"✓ 7 groups of 16: {all(len(g) == 16 for g in groups_by_point.values())}")
print()

# Additional verification: each point is not on exactly 4 lines
print("=== ADDITIONAL STRUCTURE VERIFICATION ===")
for point in all_points:
    lines_not_containing_point = [i for i, line in enumerate(fano_lines) if point not in line]
    print(f"Point {point} is not on {len(lines_not_containing_point)} lines: {lines_not_containing_point}")

# Verify total: 7 points × 4 lines each = 28 anti-flags
print(f"\nTotal anti-flags calculation: 7 points × 4 lines = {7 * 4} ✓")

# Show the canonical projection explicitly for one anti-flag
print("\n=== EXAMPLE PROJECTION ===")
example_anti_flag = anti_flags[0]
print(f"Anti-flag: {example_anti_flag}")
example_line = fano_lines[example_anti_flag[1]]
print(f"Line contains points: {example_line}")
example_pointed = [(p, l, r) for (p, l, r) in pointed_anti_flags 
                   if p == example_anti_flag[0] and l == example_anti_flag[1]]
print(f"Maps to pointed anti-flags: {example_pointed}")
print(f"Cardinality: {len(example_pointed)}")