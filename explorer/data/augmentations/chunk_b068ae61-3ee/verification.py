import numpy as np
from itertools import combinations

# Define the Fano plane as 7 points and 7 lines
# Points: 0, 1, 2, 3, 4, 5, 6
# Lines defined by their point sets
fano_lines = [
    {0, 1, 2},  # Line 0
    {0, 3, 5},  # Line 1
    {0, 4, 6},  # Line 2
    {1, 3, 6},  # Line 3
    {1, 4, 5},  # Line 4
    {2, 3, 4},  # Line 5
    {2, 5, 6}   # Line 6
]

# Fix p₀ = 0 as our stabilizing point
p0 = 0

print("=== FANO PLANE STRUCTURE ===")
print(f"Fixed point p₀ = {p0}")
print("\nAll lines in Fano plane:")
for i, line in enumerate(fano_lines):
    print(f"Line {i}: {sorted(line)}")

# Find lines passing through p₀ and lines not passing through p₀
lines_through_p0 = []
lines_not_through_p0 = []

for i, line in enumerate(fano_lines):
    if p0 in line:
        lines_through_p0.append((i, line))
    else:
        lines_not_through_p0.append((i, line))

print(f"\nLines through p₀: {[l[0] for l in lines_through_p0]}")
print(f"Lines NOT through p₀ (Muladhara 4): {[l[0] for l in lines_not_through_p0]}")

# Generate all permutations of {0,1,2,3,4,5,6} that fix p₀
# These form the stabilizer subgroup Stab(p₀)
def apply_permutation(perm, line_set):
    """Apply permutation to a line (set of points)"""
    return {perm[p] for p in line_set}

def is_valid_fano_automorphism(perm):
    """Check if permutation preserves Fano plane structure"""
    # Apply permutation to all lines
    permuted_lines = [apply_permutation(perm, line) for line in fano_lines]
    # Check if we get the same set of lines (in any order)
    return set(frozenset(line) for line in permuted_lines) == set(frozenset(line) for line in fano_lines)

# Generate stabilizer subgroup: permutations fixing p₀ that preserve Fano structure
print("\n=== STABILIZER SUBGROUP Stab(p₀) ===")
stabilizer = []

# We'll check permutations that fix 0 and permute {1,2,3,4,5,6}
from itertools import permutations

count = 0
for perm_tuple in permutations(range(1, 7)):
    # Create full permutation fixing 0
    perm = [0] + list(perm_tuple)
    perm_dict = {i: perm[i] for i in range(7)}
    
    if is_valid_fano_automorphism(perm_dict):
        stabilizer.append(perm_dict)
        count += 1
        if count <= 3:  # Show first few examples
            print(f"Element {count}: {perm}")

print(f"\nStabilizer size: |Stab(p₀)| = {len(stabilizer)}")

# Now check the orbit structure of the 4 lines not through p₀
print("\n=== ORBIT ANALYSIS ===")
print("Checking if the 4 lines not through p₀ form a single orbit...")

# For each pair of lines not through p₀, check if there's a stabilizer element mapping one to the other
orbit_connections = {}
for i, (idx1, line1) in enumerate(lines_not_through_p0):
    orbit_connections[idx1] = []
    for j, (idx2, line2) in enumerate(lines_not_through_p0):
        if i != j:
            # Check if any stabilizer element maps line1 to line2
            for g in stabilizer:
                if apply_permutation(g, line1) == line2:
                    orbit_connections[idx1].append(idx2)
                    break

print("\nOrbit connectivity (line -> reachable lines):")
for line_idx, reachable in orbit_connections.items():
    print(f"Line {line_idx} -> Lines {reachable}")

# Verify single orbit by checking if all 4 lines are reachable from any starting line
def compute_orbit(start_line_idx, lines_not_through_p0_dict):
    """Compute full orbit of a line under stabilizer action"""
    orbit = {start_line_idx}
    to_check = [start_line_idx]
    
    while to_check:
        current = to_check.pop()
        for g in stabilizer:
            current_line = lines_not_through_p0_dict[current]
            image_line = apply_permutation(g, current_line)
            # Find which line index this is
            for idx, line in lines_not_through_p0_dict.items():
                if image_line == line and idx not in orbit:
                    orbit.add(idx)
                    to_check.append(idx)
    
    return orbit

# Create dictionary for easier lookup
lines_not_through_p0_dict = {idx: line for idx, line in lines_not_through_p0}

# Compute orbit starting from first line not through p₀
start_line = lines_not_through_p0[0][0]
orbit = compute_orbit(start_line, lines_not_through_p0_dict)

print(f"\nOrbit of line {start_line}: {sorted(orbit)}")
print(f"Orbit size: {len(orbit)}")
print(f"Number of lines not through p₀: {len(lines_not_through_p0)}")

# Verify it's a single orbit
assert len(orbit) == len(lines_not_through_p0), "The 4 lines do not form a single orbit!"
print("\n✓ VERIFIED: The 4 lines not through p₀ form a SINGLE orbit under Stab(p₀)")

# Additional verification: show the symmetry explicitly
print("\n=== SYMMETRY DEMONSTRATION ===")
print("Showing how stabilizer elements permute the Muladhara 4:")
# Pick a few stabilizer elements and show their action
for i, g in enumerate(stabilizer[:3]):
    print(f"\nStabilizer element {i+1}:")
    for idx, line in lines_not_through_p0:
        image = apply_permutation(g, line)
        # Find which line this maps to
        for idx2, line2 in lines_not_through_p0:
            if image == line2:
                if idx != idx2:
                    print(f"  Line {idx} {sorted(line)} → Line {idx2} {sorted(line2)}")
                break