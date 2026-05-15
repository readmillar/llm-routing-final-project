# Formulation Map

A1 uses `x[p,m]` only for observed pairs `(p,m) in E`.
A2 and A3 use `z[p,a]` only for prompt-feasible cascades `a in A_p`.
Single-stage cascades have depth 1 and escalation 0, so A2/A3 contain A1 as a special case when production and robustness constraints are inactive.

For a two-stage cascade `(m1,m2)`, the default linear parameters are:

`R[p,a] = r[p,m1] + (1-r[p,m1]) * rho * r[p,m2]`
`C[p,a] = c[p,m1] + (1-r[p,m1]) * c[p,m2]`
`Esc[p,a] = 1-r[p,m1]`
