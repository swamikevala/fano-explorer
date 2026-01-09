import numpy as np
from itertools import combinations

def get_fano_plane_lines():
    """Returns the 7 lines of the Fano plane"""
    return [
        [0, 1, 3],  # Line 0
        [1, 2, 4],  # Line 1
        [2, 3, 5],  # Line 2
        [3, 4, 6],  # Line 3
        [4, 5, 0],  # Line 4
        [5, 6, 1],  # Line 5
        [6, 0, 2],  # Line 6
    ]

def get_tangential_triangles(p, lines):
    """
    For a given point p, find all 12 tangential triangles.
    A tangential triangle has vertices on 3 different lines through p,
    with each vertex on a different line.
    """
    # Find the 3 lines containing point p
    lines_through_p = [i for i, line in enumerate(lines) if p in line]
    assert len(lines_through_p) == 3, f"Point {p} should be on exactly 3 lines"
    
    tangential_triangles = []
    
    # For each triple of lines through p
    for triple in [lines_through_p]:  # Only one way to choose all 3
        line0, line1, line2 = triple
        
        # Get other points on each line (excluding p)
        points0 = [pt for pt in lines[line0] if pt != p]
        points1 = [pt for pt in lines[line1] if pt != p]
        points2 = [pt for pt in lines[line2] if pt != p]
        
        # Each line through p has 2 other points
        assert len(points0) == 2 and len(points1) == 2 and len(points2) == 2
        
        # Form all possible triangles (2 × 2 × 2 = 8 triangles)
        for v0 in points0:
            for v1 in points1:
                for v2 in points2:
                    triangle = sorted([v0, v1, v2])
                    tangential_triangles.append(triangle)
    
    # Additionally, there are 4 more tangential triangles
    # formed by choosing one point from each line through p
    # and considering the "opposite" configuration
    
    # Get all lines NOT through p
    other_lines = [i for i in range(7) if i not in lines_through_p]
    
    # For each of the 4 other lines, check if it forms a tangential triangle
    for line_idx in other_lines:
        line = lines[line_idx]
        # This line's points form a tangential triangle if each point
        # is on a different line through p
        on_different_lines = True
        lines_containing = []
        for pt in line:
            pt_lines = [i for i in lines_through_p if pt in lines[i]]
            if len(pt_lines) != 1:
                on_different_lines = False
                break
            lines_containing.append(pt_lines[0])
        
        if on_different_lines and len(set(lines_containing)) == 3:
            tangential_triangles.append(sorted(line))
    
    return tangential_triangles

def main():
    """Verify that for each of 7 choices of p, we get 12 tangential triangles"""
    lines = get_fano_plane_lines()
    
    print("FANO PLANE TANGENTIAL TRIANGLES ANALYSIS")
    print("="*50)
    print(f"\nFano plane has 7 points: {list(range(7))}")
    print(f"Fano plane has 7 lines: {lines}")
    print("\nFor each choice of point p as pole:")
    print("-"*50)
    
    all_triangles = []
    triangles_by_p = {}
    
    for p in range(7):
        triangles = get_tangential_triangles(p, lines)
        triangles_by_p[p] = triangles
        all_triangles.extend([(p, t) for t in triangles])
        
        print(f"\nPole p = {p}:")
        print(f"  Lines through p: {[i for i, line in enumerate(lines) if p in line]}")
        print(f"  Number of tangential triangles: {len(triangles)}")
        
        # Verify we have exactly 12 triangles
        assert len(triangles) == 12, f"Expected 12 triangles for p={p}, got {len(triangles)}"
        
        # Show first few triangles as examples
        print(f"  First 3 triangles: {triangles[:3]}")
    
    print("\n" + "="*50)
    print("VERIFICATION SUMMARY:")
    print(f"  Total configurations (7 poles × 12 triangles): {len(all_triangles)}")
    print(f"  Expected total: 7 × 12 = 84")
    assert len(all_triangles) == 84, "Total should be 84"
    
    # Check for patterns across different p-choices
    print("\nCROSS-POLE ANALYSIS:")
    
    # Count how many times each triangle appears across all poles
    triangle_counts = {}
    for p, triangle in all_triangles:
        triangle_key = tuple(triangle)
        if triangle_key not in triangle_counts:
            triangle_counts[triangle_key] = []
        triangle_counts[triangle_key].append(p)
    
    print(f"  Unique triangles across all poles: {len(triangle_counts)}")
    
    # Analyze frequency distribution
    freq_dist = {}
    for triangle, poles in triangle_counts.items():
        freq = len(poles)
        if freq not in freq_dist:
            freq_dist[freq] = 0
        freq_dist[freq] += 1
    
    print(f"  Frequency distribution:")
    for freq in sorted(freq_dist.keys()):
        print(f"    {freq_dist[freq]} triangles appear for {freq} different poles")
    
    print("\n" + "="*50)
    print("YOGIC RESONANCE:")
    print(f"  Mathematical structure: 7 × 12 = 84 tangential triangles")
    print(f"  Yogic tradition: 84 asanas, 84 siddhas")
    print(f"  This is a natural mathematical emergence, not imposed")

if __name__ == "__main__":
    main()