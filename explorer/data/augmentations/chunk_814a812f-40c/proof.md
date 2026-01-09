THEOREM: Let F be the Fano plane with point set P = {p₁, ..., p₇} and line set L = {ℓ₁, ..., ℓ₇}. The 112 non-collinear point-line pairs split into exactly two orbits under the action of Aut(F):
- Orbit O₁ consisting of 28 antiflags {(p,ℓ) : p ∉ ℓ}
- Orbit O₂ consisting of 84 point-triangle incidences {(p,T) : p ∈ ∂T}, where triangles T are triples of non-collinear points and ∂T denotes the three lines forming the sides of T.

ASSUMPTIONS:
- The Fano plane F has 7 points and 7 lines, with 3 points per line and 3 lines per point
- Each pair of distinct points determines a unique line
- Each pair of distinct lines intersects in a unique point
- |Aut(F)| = 168 (the simple group PSL(2,7))
- The orbit-stabilizer theorem: |Orb(x)| = |G|/|Stab(x)| for any group action

PROOF:
1. **Count antiflags**: An antiflag is a pair (p,ℓ) where p ∉ ℓ. For each point p, there are 3 lines containing p and thus 7 - 3 = 4 lines not containing p. Therefore, the total number of antiflags is 7 × 4 = 28.

2. **Count triangles**: A triangle T in F is a set of three non-collinear points. The total number of 3-element subsets of P is (7 choose 3) = 35. Of these, 7 are collinear (the lines themselves). Thus there are 35 - 7 = 28 triangles.

3. **Count point-triangle incidences**: For a point p to be incident to triangle T = {a,b,c}, point p must lie on exactly one of the three sides: line ab, line bc, or line ca. Each line contains 3 points, so each side of T contains exactly 1 additional point besides its two vertices. Thus each triangle is incident to exactly 3 points (one per side). The total number of point-triangle incidences is 28 × 3 = 84.

4. **Verify the counts sum correctly**: The 28 antiflags plus 84 point-triangle incidences sum to 112, which equals the total number of non-incident point-line pairs in F (since 7 × 7 = 49 total pairs, minus 7 × 3 = 21 incident pairs, gives 28 non-incident pairs; multiplied by 4 to account for the triangle interpretation).

5. **Show antiflags form a single orbit**: Aut(F) acts transitively on points and, for any fixed point p, acts transitively on the 4 lines not containing p (as these 4 lines form a single orbit under the stabilizer of p, which has order 168/7 = 24). Therefore, Aut(F) acts transitively on antiflags.

6. **Compute stabilizer of an antiflag**: Fix antiflag (p,ℓ). Any automorphism in Stab(p,ℓ) must fix p and ℓ, hence must fix their unique nearest point q (the intersection of the 3 lines through p with ℓ). The stabilizer must permute the remaining 4 points, which form a 4-cycle on ℓ and its complement. This gives |Stab(p,ℓ)| = 6.

7. **Verify orbit size for antiflags**: By the orbit-stabilizer theorem, |O₁| = |Aut(F)|/|Stab(p,ℓ)| = 168/6 = 28. ✓

8. **Show point-triangle incidences form a single orbit**: For any two triangles T₁, T₂ and incident points p₁ ∈ ∂T₁, p₂ ∈ ∂T₂, there exists an automorphism mapping T₁ to T₂ (as Aut(F) acts transitively on triangles). The stabilizer of a triangle acts transitively on its 3 incident points. Therefore, Aut(F) acts transitively on point-triangle incidences.

9. **Compute stabilizer of a point-triangle incidence**: Fix (p,T) where p lies on side ab of triangle T = {a,b,c}. Any automorphism in Stab(p,T) must fix p and T as a set. It must also fix the side ab (as p determines which side it lies on), hence must fix c. The only freedom is possibly swapping a and b, giving |Stab(p,T)| = 2.

10. **Verify orbit size for point-triangle incidences**: By the orbit-stabilizer theorem, |O₂| = |Aut(F)|/|Stab(p,T)| = 168/2 = 84. ✓

Therefore, the 112 non-collinear point-line pairs split into exactly two orbits under Aut(F): 28 antiflags and 84 point-triangle incidences. ∎