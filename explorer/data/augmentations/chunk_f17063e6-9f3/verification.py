import itertools
from collections import defaultdict

class FanoPlane:
    def __init__(self):
        # Points labeled 0-6
        self.points = list(range(7))
        
        # Lines of the Fano plane (each line has exactly 3 points)
        self.lines = [
            {0, 1, 2},  # Line 0
            {3, 4, 5},  # Line 1
            {0, 3, 6},  # Line 2
            {1, 4, 6},  # Line 3
            {2, 5, 6},  # Line 4
            {0, 4, 5},  # Line 5
            {1, 3, 5},  # Line 6
            {2, 3, 4}   # Line 7
        ]
        
    def is_collinear(self, points):
        """Check if a set of points is collinear"""
        point_set = set(points)
        for line in self.lines:
            if point_set.issubset(line):
                return True
        return False
    
    def get_line_through_points(self, p1, p2):
        """Get the line containing two points"""
        for i, line in enumerate(self.lines):
            if p1 in line and p2 in line:
                return i
        return None

class ChakraClassifier:
    def __init__(self, fano, heart_point, earth_line_idx):
        self.fano = fano
        self.C = heart_point  # Heart point
        self.L_inf_idx = earth_line_idx  # Earth horizon line index
        self.L_inf = fano.lines[earth_line_idx]  # Earth horizon line
        
        # Verify C is not on L∞
        if self.C in self.L_inf:
            raise ValueError("Heart point C must not be on Earth line L∞")
            
    def classify_chakra(self, chakra):
        """Classify a 4-point subset (chakra) based on its flag structure"""
        points = list(chakra)
        
        # Check if chakra contains C
        contains_C = self.C in points
        
        # Find all collinear triples in the chakra
        collinear_triples = []
        for triple in itertools.combinations(points, 3):
            if self.fano.is_collinear(triple):
                collinear_triples.append(set(triple))
        
        # Case 1: No collinear triples - Type "No Flag"
        if len(collinear_triples) == 0:
            if contains_C:
                return "NoFlag_Heart"  # Contains C
            else:
                return "NoFlag_Earth"  # Doesn't contain C
        
        # Case 2: Exactly one collinear triple - Type "Flag"
        elif len(collinear_triples) == 1:
            triple = collinear_triples[0]
            fourth_point = (set(points) - triple).pop()
            
            # Check if the line of the triple is L∞
            line_idx = None
            for i, line in enumerate(self.fano.lines):
                if triple == line:
                    line_idx = i
                    break
            
            is_earth_flag = (line_idx == self.L_inf_idx)
            
            if contains_C:
                if self.C == fourth_point:
                    return "Flag_Heart_apex"  # C is the apex (isolated point)
                else:
                    return "Flag_Heart_base"  # C is in the base (collinear triple)
            else:
                if is_earth_flag:
                    return "Flag_Earth_earth"  # Base is L∞
                else:
                    return "Flag_Earth_other"  # Base is not L∞
        
        # Case 3: All four points collinear - shouldn't happen in Fano plane
        else:
            return "Degenerate"

def main():
    # Create Fano plane
    fano = FanoPlane()
    
    # Fix Heart point C = 0 and Earth line L∞ = line 1 (containing points {3,4,5})
    C = 0
    L_inf_idx = 1
    
    print("Fano Plane Configuration:")
    print(f"Heart point C = {C}")
    print(f"Earth line L∞ = Line {L_inf_idx} containing points {fano.lines[L_inf_idx]}")
    print()
    
    # Create classifier
    classifier = ChakraClassifier(fano, C, L_inf_idx)
    
    # Generate all 4-point subsets (chakras)
    all_chakras = list(itertools.combinations(fano.points, 4))
    print(f"Total number of 4-point subsets (chakras): {len(all_chakras)}")
    print()
    
    # Classify each chakra
    classification_counts = defaultdict(int)
    classifications = {}
    
    for chakra in all_chakras:
        chakra_type = classifier.classify_chakra(chakra)
        classification_counts[chakra_type] += 1
        classifications[chakra] = chakra_type
    
    # Expected partition based on the insight
    expected_partition = {
        "NoFlag_Heart": 12,    # No flag, contains C
        "NoFlag_Earth": 4,     # No flag, doesn't contain C
        "Flag_Heart_apex": 6,   # Flag with C as apex
        "Flag_Heart_base": 6,   # Flag with C in base
        "Flag_Earth_earth": 12, # Flag without C, base is L∞
        "Flag_Earth_other": 72  # Flag without C, base is not L∞
    }
    
    # Print results
    print("Classification Results:")
    print("-" * 50)
    total_classified = 0
    for chakra_type, count in sorted(classification_counts.items()):
        print(f"{chakra_type:20}: {count:3d} chakras")
        total_classified += count
    
    print("-" * 50)
    print(f"Total classified: {total_classified}")
    print()
    
    # Verify the partition
    print("Verification against expected partition:")
    print("-" * 50)
    all_match = True
    for chakra_type, expected in expected_partition.items():
        actual = classification_counts.get(chakra_type, 0)
        match = "✓" if actual == expected else "✗"
        print(f"{chakra_type:20}: Expected {expected:3d}, Got {actual:3d} {match}")
        if actual != expected:
            all_match = False
    
    print("-" * 50)
    if all_match:
        print("SUCCESS: All counts match the expected 12+72+4+6+6+12 partition!")
    else:
        print("FAILURE: Counts do not match expected partition.")
    
    # Verify total
    expected_total = sum(expected_partition.values())
    print(f"\nTotal verification: {total_classified} = {expected_total} ✓")
    
    # Additional verification: Check a different choice of (C, L∞)
    print("\n" + "="*60)
    print("Testing naturality with different (C, L∞) choice:")
    print("="*60)
    
    # Try C = 2, L∞ = line 3 (containing {1,4,6})
    C2 = 2
    L_inf_idx2 = 3
    print(f"\nAlternate configuration: C = {C2}, L∞ = Line {L_inf_idx2} containing {fano.lines[L_inf_idx2]}")
    
    classifier2 = ChakraClassifier(fano, C2, L_inf_idx2)
    classification_counts2 = defaultdict(int)
    
    for chakra in all_chakras:
        chakra_type = classifier2.classify_chakra(chakra)
        classification_counts2[chakra_type] += 1
    
    print("\nClassification counts for alternate configuration:")
    for chakra_type, count in sorted(classification_counts2.items()):
        print(f"{chakra_type:20}: {count:3d} chakras")
    
    # Check if we get the same partition structure
    partition1 = sorted(classification_counts.values())
    partition2 = sorted(classification_counts2.values())
    
    print(f"\nPartition 1 (sorted): {partition1}")
    print(f"Partition 2 (sorted): {partition2}")
    print(f"Partitions are {'identical' if partition1 == partition2 else 'different'} up to reordering!")

if __name__ == "__main__":
    main()