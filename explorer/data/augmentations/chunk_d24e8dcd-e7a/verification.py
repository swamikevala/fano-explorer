import numpy as np
from itertools import combinations, permutations

# Define the 7 points of the Fano plane
points = list(range(7))

# Define the 7 lines of the Fano plane (each line contains 3 points)
# Using the standard coordinatization
lines = [
    (0, 1, 3),  # Line 1
    (1, 2, 4),  # Line 2
    (2, 3, 5),  # Line 3
    (3, 4, 6),  # Line 4
    (4, 5, 0),  # Line 5
    (5, 6, 1),  # Line 6
    (6, 0, 2)   # Line 7
]

# Generate all 28 triangles (3-element subsets of points)
all_triangles = list(combinations(points, 3))
print(f"Total number of triangles: {len(all_triangles)}")

# Separate triangles into collinear (lines) and non-collinear
collinear_triangles = [tuple(sorted(line)) for line in lines]
non_collinear_triangles = [t for t in all_triangles if t not in collinear_triangles]

print(f"Collinear triangles (lines): {len(collinear_triangles)}")
print(f"Non-collinear triangles: {len(non_collinear_triangles)}")
print()

# For each point, find all triangles not containing it
anti_incidences = {}
for point in points:
    triangles_without_point = [t for t in all_triangles if point not in t]
    anti_incidences[point] = triangles_without_point
    print(f"Point {point}: {len(triangles_without_point)} triangles not containing it")

# Verify total count is 112
total_anti_incidences = sum(len(triangles) for triangles in anti_incidences.values())
print(f"\nTotal anti-incidence pairs: {total_anti_incidences}")
assert total_anti_incidences == 112, f"Expected 112, got {total_anti_incidences}"

# Define automorphism generators for the Fano plane
# Generator 1: cyclic permutation (0 1 2 3 4 5 6)
def cyclic_perm(x):
    return (x + 1) % 7

# Generator 2: Frobenius automorphism (specific to this representation)
# This is one possible choice that generates the full group with cyclic_perm
def frobenius_perm(x):
    # Map according to the pattern that preserves the Fano structure
    mapping = {0: 0, 1: 2, 2: 4, 3: 6, 4: 1, 5: 3, 6: 5}
    return mapping[x]

# Generate some automorphisms by composing the generators
def apply_perm(triangle, perm_func):
    return tuple(sorted(perm_func(p) for p in triangle))

# Check transitivity: for any two anti-incidence pairs, there exists an automorphism
print("\nChecking transitivity of automorphism action...")

# Test a few examples
test_pairs = [
    ((0, (1, 2, 4)), (3, (0, 1, 5))),  # Two different anti-incidence pairs
    ((1, (0, 3, 6)), (5, (1, 2, 4))),
]

for (p1, t1), (p2, t2) in test_pairs:
    print(f"\nTesting transformation from ({p1}, {t1}) to ({p2}, {t2})")
    
    # Try to find an automorphism mapping p1 to p2
    # For demonstration, we'll check a few composed automorphisms
    found = False
    for i in range(7):  # Apply cyclic i times
        transformed_p1 = p1
        for _ in range(i):
            transformed_p1 = cyclic_perm(transformed_p1)
        
        if transformed_p1 == p2:
            # Check if the triangle transforms correctly
            transformed_t1 = t1
            for _ in range(i):
                transformed_t1 = apply_perm(transformed_t1, cyclic_perm)
            
            print(f"  Cyclic^{i} maps point {p1} → {p2}")
            print(f"  Triangle {t1} → {transformed_t1}")
            
            # Verify it's still an anti-incidence
            if p2 not in transformed_t1:
                print(f"  ✓ Anti-incidence preserved")
                found = True
                break
    
    if not found:
        print("  (More complex automorphism needed)")

# Analyze stabilizer structure for a specific point
print("\nAnalyzing stabilizer structure for point 0...")
triangles_without_0 = anti_incidences[0]

# Group triangles by their relationship to lines through point 0
lines_through_0 = [line for line in lines if 0 in line]
print(f"Lines through point 0: {lines_through_0}")

# Categorize the 16 triangles
categories = {
    "disjoint_from_all_lines": [],
    "intersects_one_line": [],
    "intersects_two_lines": [],
    "intersects_three_lines": []
}

for triangle in triangles_without_0:
    intersection_count = 0
    for line in lines_through_0:
        if len(set(triangle) & set(line)) > 0:
            intersection_count += 1
    
    if intersection_count == 0:
        categories["disjoint_from_all_lines"].append(triangle)
    elif intersection_count == 1:
        categories["intersects_one_line"].append(triangle)
    elif intersection_count == 2:
        categories["intersects_two_lines"].append(triangle)
    else:
        categories["intersects_three_lines"].append(triangle)

print("\nPartition of 16 triangles not containing point 0:")
for category, triangles in categories.items():
    print(f"  {category}: {len(triangles)} triangles")
    if len(triangles) <= 4:  # Only print if small number
        for t in triangles:
            print(f"    {t}")