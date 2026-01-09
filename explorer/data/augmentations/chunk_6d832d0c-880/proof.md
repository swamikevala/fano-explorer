THEOREM: Let F be the Fano plane with point set P and line set L. The set of pointed anti-flags T = {(p,l,q) : p ∈ P, l ∈ L, p ∉ l, q ∈ l} has cardinality 84, and the projection π: T → A defined by π(p,l,q) = (p,l) onto the set of anti-flags A = {(p,l) : p ∈ P, l ∈ L, p ∉ l} is a 3-to-1 surjection with |A| = 28.

ASSUMPTIONS:
- F is the Fano plane with |P| = 7 and |L| = 7
- Each line contains exactly 3 points
- Each point lies on exactly 3 lines
- Any two distinct points determine a unique line
- Any two distinct lines intersect in a unique point

PROOF:
1. **Count of anti-flags |A| = 28**
   - Total ordered pairs (p,l) = |P| × |L| = 7 × 7 = 49
   - For each point p, there are exactly 3 lines containing p (by assumption)
   - Number of incident pairs {(p,l) : p ∈ l} = 7 × 3 = 21
   - Therefore |A| = 49 - 21 = 28

2. **Alternative verification of |A| = 28**
   - For each point p, there are 7 - 3 = 4 lines not containing p
   - Sum over all points: 7 × 4 = 28
   - This confirms |A| = 28

3. **Well-definedness of π**
   - For any (p,l,q) ∈ T, we have p ∉ l by definition
   - Thus (p,l) ∈ A, so π is well-defined

4. **Fiber size |π⁻¹(p,l)| = 3 for each (p,l) ∈ A**
   - Fix (p,l) ∈ A, so p ∉ l
   - π⁻¹(p,l) = {(p,l,q) : q ∈ l}
   - Since l contains exactly 3 points, |π⁻¹(p,l)| = 3

5. **Surjectivity of π**
   - For any (p,l) ∈ A, choose any q ∈ l
   - Then (p,l,q) ∈ T and π(p,l,q) = (p,l)
   - Therefore π is surjective

6. **Count of pointed anti-flags |T| = 84**
   - By steps 3-5, π is a 3-to-1 surjection
   - Therefore |T| = 3 × |A| = 3 × 28 = 84

7. **Conclusion**
   - We have shown |A| = 28 (stationary nadis)
   - We have shown |T| = 84 (flowing nadis)
   - The projection π: T → A is 3-to-1, establishing the structural relationship ∎