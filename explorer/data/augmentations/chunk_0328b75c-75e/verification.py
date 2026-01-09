import itertools
from collections import defaultdict

# Define the Fano plane
# Points: 7 points labeled 0-6
points = list(range(7))

# Lines: 7 lines, each containing 3 points
# These satisfy the Fano plane axioms
lines = [
    [0, 1, 3],  # Line 0
    [1, 2, 4],  # Line 1
    [2, 3, 5],  # Line 2
    [3, 4, 6],  # Line 3
    [4, 5, 0],  # Line 4
    [5, 6, 1],  # Line 5
    [6, 0, 2],  # Line 6
]

print("=== FANO PLANE STRUCTURE ===")
print(f"Points: {points}")
print(f"Lines:")
for i, line in enumerate(lines):
    print(f"  Line {i}: {line}")

# Verify Fano plane properties
print("\n=== VERIFYING FANO PLANE PROPERTIES ===")
# Check: each pair of points lies on exactly one line
for p1, p2 in itertools.combinations(points, 2):
    lines_containing_both = [line for line in lines if p1 in line and p2 in line]
    assert len(lines_containing_both) == 1, f"Points {p1},{p2} should be on exactly 1 line"
print("✓ Each pair of points lies on exactly one line")

# Generate all triangles (non-collinear triples)
print("\n=== GENERATING TRIANGLES ===")
triangles = []
for triple in itertools.combinations(points, 3):
    # Check if triple is non-collinear (not all on same line)
    is_collinear = any(all(p in line for p in triple) for line in lines)
    if not is_collinear:
        triangles.append(sorted(triple))

triangles.sort()
print(f"Number of triangles: {len(triangles)}")
print(f"First 5 triangles: {triangles[:5]}")

# For each triangle, verify it's non-collinear
for triangle in triangles:
    assert not any(all(p in line for p in triangle) for line in lines), \
        f"Triangle {triangle} should be non-collinear"
print("✓ All triangles verified as non-collinear")

# Generate all anti-incidences (x, T) where x is not in triangle T
print("\n=== COMPUTING ANTI-INCIDENCES ===")
anti_incidences = []
anti_incidences_by_witness = defaultdict(list)

for x in points:
    for T in triangles:
        if x not in T:
            anti_incidences.append((x, tuple(T)))
            anti_incidences_by_witness[x].append(tuple(T))

print(f"Total anti-incidences: {len(anti_incidences)}")

# Verify the 7×16 structure
print("\n=== VERIFYING 7×16 STRUCTURE ===")
for x in points:
    count = len(anti_incidences_by_witness[x])
    print(f"Witness point {x}: {count} anti-incident triangles")
    assert count == 16, f"Point {x} should have exactly 16 anti-incident triangles"

# Verify total count
total = sum(len(anti_incidences_by_witness[x]) for x in points)
assert total == 112, f"Total should be 112, got {total}"
assert total == 7 * 16, "Total should equal 7 × 16"
print(f"\nTotal verification: 7 × 16 = {total} ✓")

# Show witness-process duality structure
print("\n=== WITNESS-PROCESS DUALITY ===")
# For a sample point, show its anti-incident triangles
sample_witness = 0
print(f"Sample witness point {sample_witness} sees these 16 triangles:")
for i, T in enumerate(anti_incidences_by_witness[sample_witness]):
    print(f"  {i+1:2d}. Triangle {list(T)}")

# Show that each triangle is seen by exactly 4 witnesses
print("\n=== DUAL PERSPECTIVE ===")
triangle_to_witnesses = defaultdict(list)
for x, T in anti_incidences:
    triangle_to_witnesses[T].append(x)

sample_triangle = triangles[0]
witnesses = triangle_to_witnesses[tuple(sample_triangle)]
print(f"Sample triangle {sample_triangle} is seen by witnesses: {sorted(witnesses)}")
print(f"Number of witnesses: {len(witnesses)}")

# Verify each triangle has exactly 4 witnesses
for T in triangles:
    assert len(triangle_to_witnesses[tuple(T)]) == 4, \
        f"Triangle {T} should have exactly 4 witnesses"
print("✓ Each triangle has exactly 4 witnesses")

print("\n=== FINAL VERIFICATION ===")
print(f"112 chakras = {len(points)} witnesses × {len(anti_incidences_by_witness[0])} triangles per witness")
print(f"112 chakras = {len(triangles)} triangles × {len(triangle_to_witnesses[tuple(triangles[0])])} witnesses per triangle")
print("✓ Both decompositions yield 112")