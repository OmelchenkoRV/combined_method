This method generates a standard structure from an Excel file and solves subproblems for providers and IT companies using a probabilistic approach. Then, the Improvement is based on a heuristic procedure for the total solution of subproblems.
Here is a breakdown of the method by steps:
Stage 1. Solving the subproblem for an IT company based on the probabilistic method. The most profitable package of services for the IT company is formed by observing all constraints – resource, provider, and inter-service dependency.
Stage 2. Solving the subproblem for provider j (j=1,…,m) based on the probabilistic method. The most profitable package of services for each provider is formed, taking into account all constraints – resource, provider, and inter-service dependency.
Stage 3. Searching for the solution to the corresponding subproblem for providers based on the solution to the subproblem for the IT company obtained at stage 1.
Stage 4. Searching for the solution to the corresponding subproblem for the IT company based on the solutions of the subproblem for providers obtained at stage 2.
Stage 5. Improvement based on the heuristic procedure of the total solution of subproblems for the IT company and for providers j (j=1,…,m). Mutually beneficial discounts are determined for both parties.
Stage 6. Improvement based on the heuristic procedure of the total solution of subproblems for providers j (j=1,…,m) and for the IT company. Mutually beneficial discounts are determined for both parties.
Stage 7. Selection of the best solution from stages 5 and 6.

Subtask for an IT company
Criteria: maximisation of the objective function (1.1):
W=∑_(j=1)^k▒∑_(i=1)^m▒〖d_ij (1-r_ij ) v_ij.〗
Constraints (1.3) - (1.6):
∑_(j=1)^k▒∑_(i=1)^m▒〖β_ijl v〗_ijl ≤T_l,
s_ij v_ij≤  p_ij/b_ij ,i=1,…,k,j=1,…,m,
a_ijg v_ij≤a_ig,i=1,…,k,j=1,…,m,g=1,…,G,
v_ij v_lj=_il,i=1,…,k,j=1,…,m,l=1,…,k.
Subtask for providers
Criteria: maximisation of the objective function (1.2):
Q_j=∑_(i=1)^m▒〖(p_ij-d_ij (1-r_ij)〖)v〗_ij 〗.
Constraints (1.3) - (1.6):
∑_(j=1)^k▒∑_(i=1)^m▒〖β_ijl v〗_ijl ≤T_l,
s_ij v_ij≤  p_ij/b_ij ,i=1,…,k,j=1,…,m,
a_ijg v_ij≤a_ig,i=1,…,k,j=1,…,m,g=1,…,G,
v_ij v_lj=_il,i=1,…,k,j=1,…,m,l=1,…,k.

Data structure:
class CalculationRequest(BaseModel):
    price: List[float]
    support_cost: List[float]
    daily_orders: List[List[float]]
    n_days: int
