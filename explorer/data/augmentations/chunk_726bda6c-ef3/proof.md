THEOREM: In the Fano plane, the 112 anti-flags (ordered pairs (p,T) where p is a point not in triangle T) partition into exactly 84 "conductive" anti-flags (where p lies on at least one edge-line of T) and 28 "radical" anti-flags (where p lies on no edge-line of T).

ASSUMPTIONS:
- The Fano plane has 7 points and 7 lines
- Each line contains exactly 3 points
- Each point lies on exactly 3 lines
- Any two distinct points determine a unique line
- A triangle T is a set of 3 non-collinear points
- An anti-flag is a pair (p,T) where p ∉ T

PROOF:
1. The Fano plane contains exactly 28 triangles.
   [Justification: There are C(7,3) = 35 3-subsets of points. Each line accounts for 1 collinear triple. With 7 lines, we have 35 - 7 = 28 triangles.]

2. For each triangle T and each point p ∉ T, the pair (p,T) forms an anti-flag. Since |T| = 3, there are 7 - 3 = 4 points not in T.
   [Justification: Basic counting.]

3. The total number of anti-flags is 28 × 4 = 112.
   [Justification: Each of 28 triangles pairs with 4 external points.]

4. For any triangle T = {a,b,c}, the edge-lines are L_ab (containing a,b), L_ac (containing a,c), and L_bc (containing b,c).
   [Justification: Definition of edge-lines.]

5. Each edge-line contains exactly one point not in T.
   [Justification: Each line has 3 points; 2 are vertices of T, so 1 point remains.]

6. The three additional points on the edge-lines are distinct.
   [Justification: If two edge-lines shared a point p ∉ T, then p would lie on two lines through distinct pairs of vertices of T. But in the Fano plane, two distinct lines intersect in at most one point, and these lines already intersect at vertices of T.]

7. The union of the three edge-lines contains exactly 6 points: the 3 vertices of T plus 3 distinct additional points.
   [Justification: From steps 5 and 6.]

8. Since the Fano plane has 7 points total and 6 lie on edge-lines of T, exactly 1 point lies on no edge-line of T.
   [Justification: 7 - 6 = 1.]

9. For each triangle T, exactly 3 of the 4 external points are "conductive" (lie on edge-lines) and exactly 1 is "radical" (lies on no edge-line).
   [Justification: From steps 7 and 8.]

10. The number of conductive anti-flags is 28 × 3 = 84.
    [Justification: Each of 28 triangles contributes 3 conductive anti-flags.]

11. The number of radical anti-flags is 28 × 1 = 28.
    [Justification: Each of 28 triangles contributes 1 radical anti-flag.]

12. We verify: 84 + 28 = 112, confirming our partition.
    [Justification: Arithmetic check.] ∎