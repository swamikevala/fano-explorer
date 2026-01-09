import numpy as np
from itertools import combinations

def generate_fano_plane():
    """Generate the 7 points and 7 lines of the Fano plane"""
    # Points: 0,1,2,3,4,5,6
    points = list(range(7))
    
    # Lines in the Fano plane (each line contains 3 points)
    lines = [
        (0, 1, 2),  # Line 0
        (0, 3, 4),  # Line 1
        (0, 5, 6),  # Line 2
        (1, 3, 5),  # Line 3
        (1, 4, 6),  # Line 4
        (2, 3, 6),  # Line 5
        (2, 4, 5),  # Line 6
    ]
    
    return points, lines

def get_all_triangles(lines):
    """Generate all 28 triangles (3-line combinations) in the Fano plane"""
    triangles = []
    for combo in combinations(range(len(lines)), 3):
        triangles.append(combo)
    return triangles

def get_triangle_points(triangle_indices, lines):
    """Get the 3 vertices of a triangle given its line indices"""
    line1, line2, line3 = [set(lines[i]) for i in triangle_indices]
    
    # Each vertex is the intersection of two lines
    vertex1 = line1 & line2
    vertex2 = line1 & line3
    vertex3 = line2 & line3
    
    # Convert sets to single points
    vertices = []
    for v in [vertex1, vertex2, vertex3]:
        if len(v) == 1:
            vertices.append(list(v)[0])
    
    return vertices if len(vertices) == 3 else None

def get_external_points(triangle_points, all_points):
    """Get the 4 points not in the triangle"""
    return [p for p in all_points if p not in triangle_points]

def point_on_line(point, line):
    """Check if a point lies on a line"""
    return point in line

def classify_anti_flags():
    """Classify all 112 anti-flags as conductive or radical"""
    points, lines = generate_fano_plane()
    triangles = get_all_triangles(lines)
    
    conductive_count = 0
    radical_count = 0
    
    # Store examples for demonstration
    conductive_examples = []
    radical_examples = []
    
    print(f"Total triangles in Fano plane: {len(triangles)}")
    print(f"Points per triangle: 3")
    print(f"External points per triangle: 4")
    print(f"Total anti-flags: {len(triangles)} × 4 = {len(triangles) * 4}\n")
    
    for tri_idx, triangle_indices in enumerate(triangles):
        # Get the three lines forming this triangle
        triangle_lines = [lines[i] for i in triangle_indices]
        
        # Get the vertices of the triangle
        triangle_points = get_triangle_points(triangle_indices, lines)
        
        if triangle_points is None:
            continue
            
        # Get the 4 external points
        external_points = get_external_points(triangle_points, points)
        
        # For each external point, check if it's conductive or radical
        for ext_point in external_points:
            is_conductive = False
            
            # Check if the external point lies on any of the triangle's lines
            for line in triangle_lines:
                if point_on_line(ext_point, line):
                    is_conductive = True
                    break
            
            if is_conductive:
                conductive_count += 1
                if len(conductive_examples) < 3:
                    conductive_examples.append({
                        'triangle': triangle_indices,
                        'vertices': triangle_points,
                        'external_point': ext_point,
                        'line_containing_point': [i for i, line in enumerate(triangle_lines) 
                                                 if point_on_line(ext_point, line)]
                    })
            else:
                radical_count += 1
                if len(radical_examples) < 3:
                    radical_examples.append({
                        'triangle': triangle_indices,
                        'vertices': triangle_points,
                        'external_point': ext_point
                    })
    
    print("=== CLASSIFICATION RESULTS ===")
    print(f"Conductive anti-flags: {conductive_count}")
    print(f"Radical anti-flags: {radical_count}")
    print(f"Total anti-flags: {conductive_count + radical_count}")
    
    # Verify the split
    assert conductive_count == 84, f"Expected 84 conductive, got {conductive_count}"
    assert radical_count == 28, f"Expected 28 radical, got {radical_count}"
    
    print("\n=== EXAMPLES ===")
    print("\nConductive anti-flags (p lies on a line of T):")
    for i, example in enumerate(conductive_examples):
        print(f"\nExample {i+1}:")
        print(f"  Triangle formed by lines: {example['triangle']}")
        print(f"  Triangle vertices: {example['vertices']}")
        print(f"  External point p: {example['external_point']}")
        print(f"  Point p lies on triangle's line(s): {example['line_containing_point']}")
    
    print("\nRadical anti-flags (p lies on no line of T):")
    for i, example in enumerate(radical_examples):
        print(f"\nExample {i+1}:")
        print(f"  Triangle formed by lines: {example['triangle']}")
        print(f"  Triangle vertices: {example['vertices']}")
        print(f"  External point p: {example['external_point']}")
        print(f"  Point p does NOT lie on any of the triangle's lines")
    
    print("\n✓ VERIFICATION SUCCESSFUL: The 84/28 split is confirmed!")
    
    return conductive_count, radical_count

# Run the classification
classify_anti_flags()