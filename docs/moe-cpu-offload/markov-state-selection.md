# Choosing alpha to predict the states of a Markov chain

This study is a mathematical problem independent of the counter implementation.
The goal is to predict the next state from past observations.

## 1. D-state model

Consider $D\geq2$ states, denoted by $s_1,\ldots,s_D$.
We observe a sequence $X_t\in\{s_1,\ldots,s_D\}$, with initial distribution $\mu_0$ for $X_0$.
At each time $t\geq1$, we aim to predict $X_t$ before observing it,
using only $X_0,\ldots,X_{t-1}$.
A probabilistic prediction assigns a probability to each of the $D$ states.

A single time-invariant $D\times D$ transition matrix $T$ defines the chain:

$$
\Pr(X_{t+1}=s_k\mid X_t=s_j)=T(j,k).
$$

Its entries are nonnegative and each row sums to 1.
We assume there are no absorbing states:

$$
T(j,j)<1,\qquad j=1,\ldots,D.
$$

Every state therefore has a strictly positive probability of moving to another state.
This assumption does not guarantee irreducibility or aperiodicity:
a chain that always alternates between two states has no absorbing state but is periodic.

The prediction method below does not require convergence to a stationary distribution.

## 2. Exponential moving average

Each observation $X_t$ is associated with a vector $Y_t$ with $D$ components, explicitly defined by:

$$
Y_t(j)=
\begin{cases}
1 & \text{if } X_t=s_j,\\
0 & \text{if } X_t\neq s_j,
\end{cases}
\qquad j=1,\ldots,D.
$$

The $D$ components of $\widehat p_t$ estimate the frequencies of states $s_1,\ldots,s_D$.
For each component $j=1,\ldots,D$, the update is:

$$
\widehat p_t(j)=
\begin{cases}
\alpha\widehat p_{t-1}(j)+(1-\alpha) & \text{if } X_t=s_j,\\
\alpha\widehat p_{t-1}(j) & \text{if } X_t\neq s_j.
\end{cases}
$$

In other words, multiply all $D$ estimates by alpha, then add $1-\alpha$
to the observed state's estimate. In vector form:

$$
\widehat p_t=\alpha\widehat p_{t-1}+(1-\alpha)Y_t,\qquad 0\leq\alpha\leq1.
$$

Initialize $\widehat p_0$ with nonnegative components summing to 1.
The updates preserve these properties. The prediction of $X_t$ uses
$q_t=\widehat p_{t-1}$: update the estimate with $X_t$ only after evaluating that prediction.

## 3. Convergence as alpha tends to 1

If the chain is **irreducible and aperiodic**, it has a unique stationary distribution $\pi$.
The absence of absorbing states is not enough to guarantee these properties.

$$
\alpha\to1^-,\quad t(1-\alpha)\to+\infty
\quad\Longrightarrow\quad
\mathbb E\left[\|\widehat p_t-\pi\|_2^2\right]\to0.
$$

Alpha stays constant during each estimation run of length $t$.
The condition on $t$ removes the effect of initialization; at fixed time, $\widehat p_t$ would tend to $\widehat p_0$.

## 4. Selecting n states with at most k replacements

The set $S_t$ contains $n<D$ states. A decision made after observing $X_t$
replaces at most $k<n$ elements, but takes effect only at **t+2**.
The set $S_{t+1}$ is already fixed by the previous decision.

We do not know $T$. To capture correlations between successive states, estimate
each row with an exponential moving average with parameter $\beta$.
After a transition $X_{t-1}=s_i\to X_t=s_j$, update only that row:

$$
\widehat T_t(i,\ell)=
\begin{cases}
\beta\widehat T_{t-1}(i,\ell)+(1-\beta) & \text{if } \ell=j,\\
\beta\widehat T_{t-1}(i,\ell) & \text{otherwise},
\end{cases}
\qquad 0\leq\beta\leq1.
$$

Other previously learned rows stay unchanged: **forgetting occurs per departure from
a state, not at every global step**. Each row remains a probability distribution.
Before its first update, initialize a row with $\widehat p_{t-1}$.
For a row with no observed transition yet, use $\widehat p_t$ for prediction.
When $X_t=s_i$, the two-step score for $s_j$ is

$$
r_t(j)=(\widehat T_t^{\,2})(i,j)
=\sum_\ell\widehat T_t(i,\ell)\widehat T_t(\ell,j).
$$

Starting from the set **already committed for t+1**, prepare $S_{t+2}$:

- sort included states by increasing score;
- sort excluded states by decreasing score;
- replace paired states while the gain is strictly positive, at most $k$ times.

Do not revise the decision before it takes effect, even after observing $X_{t+1}$.
This rule maximizes retained mass according to the estimated scores for t+2,
not necessarily the true coverage or the long-term optimum.

## 5. Numerical comparison

The program [compare_markov_state_selection.py](compare_markov_state_selection.py)
compares two strategies on the **same trajectories** from a fixed matrix:

- **frozen EMA set**: after a warmup phase, retain the $n$ largest exponential moving averages,
  then leave the set unchanged;
- **replacements with estimated T, delay 2**: start from the same set, learn transitions
  during warmup and continue after every observation, applying the rule above.

Initialize the first two sets to the frozen set: the first decision takes effect
at the second measured observation. The true matrix is used only to generate trajectories
and evaluate conditional coverage, never to choose replacements.
No future state is used for selection.
The measured performance is the fraction of observations included in the set prepared before their arrival,
along with the average number of replacements actually applied during measurement.
The delay is modeled, but transfer costs and runtime execution time are not.

```bash
python docs/moe-cpu-offload/compare_markov_state_selection.py \
  --dimension 256 --capacity 96 --max-changes 2 --alpha 0.9 --beta 0.9 \
  --delay 2 --transition-model estimated \
  --warmup 2000 --steps 10000 --repeats 10 --seed 0 \
  --output /tmp/markov-state-selection.json
```

By default, the rows of $T$ are drawn from a Dirichlet distribution with parameter 0.2 per component.
The generated matrix is strictly positive and remains identical across repetitions.
Each trajectory starts from a uniformly sampled state; the initial estimate is uniform.
Warmup is excluded from measurements, and ties are broken by increasing state index.
The `--matrix file.json` option accepts a matrix as a list of rows.
The output JSON records the matrix, parameters, and results of each repetition.
The program numbers states from 0 to D-1.
The earlier experiment remains available with `--transition-model known --delay 1`.

With $D=256$, $k=2$, $\alpha=\beta=0.9$, a delay of 2,
2,000 warmup observations and 100,000 measured observations
(10 trajectories, seed 0):

| Capacity | Frozen EMA set | EMA-estimated T, delay 2 | Gain in percentage points |
|---|---:|---:|---:|
| 8 | 3.077% | 3.768% | +0.691 |
| 96 | 37.414% | 40.123% | +2.709 |
| 128 | 50.297% | 52.657% | +2.360 |
| 192 | 75.566% | 76.869% | +1.303 |

All four capacities use the same matrix and trajectories.
The method applies almost 2 replacements per observation. These results concern
this one matrix; no additional transfer cost is included.
