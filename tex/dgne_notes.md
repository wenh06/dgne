## Observations {#observations .unnumbered}

1\. Problem Formulation: the Networked Cournot Game can be described by
the following set of constrained optimization problems (ref. Equation
(5) of [@Yi_2019], note the difference in the last inequality
constraints): $$\label{eq:original_problem}
\def2{1}
\begin{array}{rl}
\min & g(x_i) = f_i(x_i, \mathbf{x}_{-i}) = c_i(x_i) - (P(Ax))^TA_ix_i \\
\text{s.t.} & x_i \in \Omega_i \subsetneq \mathbb{R}^{n_i} \\
& Ax \leqslant r
\end{array}
\qquad i = 1, \ldots, N$$ where $$\begin{aligned}
& c_i: \Omega_i \to \mathbb{R} \text{ is the local production cost function,} \\
& P: \mathbb{R}^m \to \mathbb{R}^m \text{ maps the total supply of each market to its corresponding price,}\end{aligned}$$
$A = [A_1, \ldots, A_N]$, $x = \operatorname{col}(x_1, \ldots, x_N)$,
$Ax = \sum\limits_{1 \leqslant j \leqslant N} A_jx_j$, $\Omega_i$ the
feasible set of $x_i$, usually a rectangle in the Euclidean space. The
last constraint is called the shared affine coupling constraint.

This set of optimization problems can be reformulated as unconstrained
optimization problems: $$\label{eq:reformulated_problem}
\begin{array}{rl}
\min & g_i(x_i) + \langle \lambda_i, r-Ax \rangle + \iota_{\Omega_i} (x_i), \quad \lambda_i \in \mathbb{R}_-^m
\end{array}$$ where $\iota_{\Omega_i} (x_i)$ is the indicator function
of $\Omega_i$, whose subgradient is the normal cone
$\mathcal{N}_{\Omega_i} (x_i)$ of $\Omega_i$, and
$\operatorname{prox}_{\mathcal{N}_{\Omega_i}}(x_i) = \Pi_{\Omega_i}(x_i)$.
Then applying (Douglas-Rachford? not the same) splitting, one gets the
Algorithms (Algorithm 1 and 2) proposed in [@Yi_2019].

2\. It is stated in the numerical study section of [@Yu_2017] that
"[However, we rely on (110) in the simulations to examine the numerical
performance regardless of solution
feasibility.]{style="color: red!60"}", where (110) is the constraints on
the market capacities, similar to $$\begin{aligned}
A_i x_i \leqslant r_i, ~ \sum r_i \leqslant r ~ (\text{or } \sum r_i = r).\end{aligned}$$

In the first paragraph of section 3.2 of [@Yi_2019], it is stated that
"[the shared affine coupling constraint is decomposed such that each
player only knows a local block of the constraint matrix. Notice that
$A_i$ characterizes how agent $i$ is involved in the coupling constraint
(shares the global resource), also assumed to be privately known by
player $i$. Then, the globally shared constraint $Ax \geqslant b$
couples the agents' feasible decision sets, but is not known by any
agent]{style="color: red!60"}".

::: question
**Question 1**. *What does it mean by "each player only knows a local
block of the constraint matrix"?*
:::

If one changes the original problem
[\[eq:original_problem\]](#eq:original_problem){reference-type="eqref"
reference="eq:original_problem"} to $$\label{eq:new_problem}
\def2{1}
\begin{array}{rl}
\min & g(x_i) = f_i(x_i, \mathbf{x}_{-i}) = c_i(x_i) - (P(Ax))^TA_ix_i \\
\text{s.t.} & x_i \in \Omega_i \subsetneq \mathbb{R}^{n_i} \\
& A_ix_i \leqslant r_i \\
& \sum r_i = r
\end{array}
\qquad i = 1, \ldots, N$$ Then the reformulated unconstrained problem is
$$\label{eq:reformulated_new_problem}
\begin{array}{rl}
\min & g_i(x_i) + \langle \lambda_i, r_i-A_ix_i \rangle + \langle \mu_i, r - \sum r_i \rangle + \iota_{\Omega_i} (x_i), \quad \lambda_i \in \mathbb{R}_-^m
\end{array}$$

3\. In the last paragraph of section 3.2 of [@Yi_2019], it is stated
that "[The update of $z_i$ in Algorithm 1 can be regarded as the
discrete-time integrator for the consensual errors of local multipliers,
which will ensure the consensus of $\lambda_i$
eventually]{style="color: red!60"}". Since the authors stated that "In
this work, we seek a GNE with the same Lagrangian multiplier for all the
agents, called variational GNE" in section 3.1

::: question
**Question 2**. *What is the situation for problems where the
multipliers do not need a consensus?*
:::

## Computation of Gradients {#computation-of-gradients .unnumbered}

Computation of the gradient of the function in equation (36) in
[@Yi_2019]

Consider the objective function
$$g(x_i) = f_i(x_i, \mathbf{x}_{-i}) = c_i(x_i) - (P(Ax))^T A_ix_i,$$

Let $p_i: \mathbb{R}^{n_i} \to \mathbb{R}^m$ be the function of supply
of the $i$-th company to the markets, i.e.
$p_i(x_i) = P(Ax) = P\left(\sum\limits_{1 \leqslant j \leqslant N} A_jx_j\right)$.
Then $$\begin{aligned}
\operatorname{grad} g(x_i) & = \nabla_{x_i} f_i(x_i, \mathbf{x}_{-i}) = \operatorname{grad} c_i(x_i) - \left( \dfrac{\partial \left( (p_i(x_i))^T A_ix_i \right)}{\partial (x_i)_k} \right)_{k=1}^{n_i} \\
& = \operatorname{grad} c_i(x_i) - \left( \dfrac{\partial \left( \sum\limits_{1\leqslant t \leqslant m} (p_i(x_i))_t (A_ix_i)_t \right)}{\partial (x_i)_k} \right)_{k=1}^{n_i} \\
& = \operatorname{grad} c_i(x_i) - \left( \sum\limits_{1\leqslant t \leqslant m}\left( \dfrac{\partial \left( (p_i(x_i))_t \right)}{\partial (x_i)_k} (A_ix_i)_t + \dfrac{\partial \left( (A_ix_i)_t \right)}{\partial (x_i)_k} (p_i(x_i))_t \right) \right)_{k=1}^{n_i} \\
% & = \operatorname{grad} c_i(x_i) - \left( \sum\limits_{1\leqslant t \leqslant m}\left( \left(\operatorname{Jac} (p_i(x_i)) \right)_{tk} (A_ix_i)_t + \dfrac{\partial \left( \sum\limits_{s} (A_i)_{ts} (x_i)_s \right)}{\partial (x_i)_k} (p_i(x_i))_t \right) \right)_{k=1}^{n_i} \\
& = \operatorname{grad} c_i(x_i) - \left( \sum\limits_{1\leqslant t \leqslant m}\left( \left(\operatorname{Jac} (p_i)(x_i) \right)_{tk} (A_ix_i)_t + (A_i)_{tk} (p_i(x_i))_t \right) \right)_{k=1}^{n_i} \\
& = \operatorname{grad} c_i(x_i) - \left( \langle \operatorname{Jac}(p_i)(x_i)_{[:,k]}, A_ix_i \rangle + \langle (A_i)_{[:,k]}, p_i(x_i) \rangle \right)_{k=1}^{n_i} \\
& =\operatorname{grad} c_i(x_i) - \left( \operatorname{Jac}(p_i)(x_i) \right)^T A_ix_i - A_i^T p_i(x_i),\end{aligned}$$
with
$\operatorname{Jac}(p_i)(x_i) = \left( \operatorname{Jac}(P)\left(\sum\limits_j A_jx_j\right) \right) \cdot A_i$.

When $p$ is the linear inverse demand function $P(s) = p − Ds$, where
$p, s \in \mathbb{R}^m$,
$D = \operatorname{diag}(d_1, \ldots, d_m) \in \operatorname{GL}_m(\mathbb{R})$.
Then $\operatorname{Jac}(P)(s) = -D$. Let the local production cost
functions be
$c_i(x_i) = \pi_i \left(\sum\limits_{j=1}^{n_i} [x_i]_j\right)^2 + b_i^Tx_i$,
then
$\operatorname{grad} c_i(x_i) = 2\pi_i \left(\sum\limits_{j=1}^{n_i} [x_i]_j\right) + b_i$.
Therefore, $$\begin{aligned}
\operatorname{grad} g(x_i) & = \left(2\pi_i \left(\sum\limits_{j=1}^{n_i} [x_i]_j\right) + b_i \right) + \left( A_i^TDA_ix_i - A_i^T \left( p - D\sum\limits_{1 \leqslant j \leqslant N} A_jx_j \right) \right)\end{aligned}$$

The expression of $\operatorname{Jac}(p_i)(x_i)$ corresponds to the
function `market_price_jac` in the file `python/simulation.py`, the
expression of $\operatorname{grad} g(x_i)$ corresponds to the attribute
`_objective_grad` of the `Company` class in
`python/networked_cournot_game.py`.

## Minimal Example {#minimal-example .unnumbered}

Consider the networked Cournot game where there is one market with two
companies. Let the price function of the market be $p(s) = 4 - s$ where
$s$ is the supply. Let the production cost functions for the two
companies be identical: $c(x_i) = x_i^2 + x_i$, $i = 1, 2$. Then the
objective function for the companies are $$\begin{cases}
\text{min} \ x_1^2 + x_1 - (4 - (x_1 + x_2)) x_1 \\
\text{min} \ x_2^2 + x_2 - (4 - (x_1 + x_2)) x_2 \\
\end{cases}$$ The solution is $x_1 = x_2 = 0.6.$
