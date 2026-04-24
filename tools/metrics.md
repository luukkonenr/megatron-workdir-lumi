# MoE Router Distributions: Metrics and How to Read Them

These notes accompany the analysis of routed-expert histograms produced for several
languages by an MoE model. For every layer $\ell$ and every language (dataset) $d$ we
have a vector of counts

$$\begin{aligned}
c^{(d,\ell)} &= \bigl(c^{(d,\ell)}_1, \dots, c^{(d,\ell)}_E\bigr), \quad c^{(d,\ell)}_i \ge 0,
\end{aligned}$$

where $E$ is the number of experts in that layer and $c^{(d,\ell)}_i$ is how often expert
$i$ was selected (over all top-$k$ selections, so each token contributes up to $k$ units of
mass). Normalizing within a layer gives the empirical routing distribution

$$p^{(d,\ell)}_i = \frac{c^{(d,\ell)}_i}{\sum_j c^{(d,\ell)}_j}, \qquad \sum_i p^{(d,\ell)}_i = 1.$$

Two qualitatively different questions get asked of these distributions:

1. **Imbalance.** For a fixed $(d, \ell)$, is $p^{(d,\ell)}$ close to uniform, or does it
  concentrate on a few experts?
2. **Divergence.** For a fixed $\ell$, how different are $p^{(d_1,\ell)}$ and
  $p^{(d_2,\ell)}$ across languages?

The metrics below are organized along that split.

---

## 1. Per-distribution imbalance

These are functionals of a single $p \in \Delta^{E-1}$ (the probability simplex). They
summarize *how peaked* the routing is, independent of any other language.

### 1.1 Shannon entropy and normalized entropy

$$H(p) = -\sum_{i=1}^E p_i \log p_i, \qquad \tilde{H}(p) = \frac{H(p)}{\log E} \in [0, 1].$$

- $\tilde{H} = 1$ iff $p$ is uniform; $\tilde{H} = 0$ iff a single expert receives all mass.
- The denominator $\log E$ is the maximum possible entropy on $E$ outcomes, which is
why $\tilde{H}$ is comparable across layers (and models) with different expert counts.
- Entropy is a *smooth* concentration measure: small mass shifts produce small
changes. It does not distinguish, on its own, "moderately spread mass over many
experts" from "a few clear winners with a long flat tail."

**In the plot.** A horizontal line near 1 across all layers means the router behaves nearly
uniformly. Dips correspond to layers where the model has decided to specialize. If the
dips at the same layer indices appear for *all* languages, the specialization is
language-agnostic; if they appear only for some languages, that layer is doing
language-specific work.

### 1.2 Effective number of experts

$$\mathrm{eff}(p) = \exp H(p) \in [1, E].$$

This is just the entropy expressed on the *count* scale. If $\mathrm{eff}(p) = 8$ on a
layer with $E = 128$, the layer is, in an information-theoretic sense, behaving as if it
had only ~8 active experts even though 128 are wired. It is the most directly
interpretable form of entropy when communicating with non-statisticians.

### 1.3 Top-$k$ mass

$$T_k(p) = \sum_{i \in \mathrm{top}_k(p)} p_i,$$

where $\mathrm{top}_k(p)$ is the index set of the $k$ largest entries of $p$.

- $T_1$ is just $\max_i p_i$; $T_8$ tells you what fraction of routing decisions are
absorbed by the busiest 8 experts.
- Intuition is direct: "60% of all tokens at this layer are sent to just 8 of the 128
experts."
- Unlike entropy, $T_k$ is sensitive only to the head of the distribution. It can disagree
with entropy in interesting ways: a distribution may have low entropy because it has
a long, slowly decaying body, even if no single expert dominates.

**In the plot.** $T_8$ versus layer is a useful companion to $\tilde{H}$. When the two
agree (low $\tilde{H}$ and high $T_8$), the layer is genuinely specializing on a small
expert set. When they disagree, the imbalance is more diffuse.

### 1.4 Gini coefficient

$$\begin{aligned}
G(p) = \frac{2 \sum_{i=1}^E i p_{(i)}}{E \sum_j p_j} - \frac{E+1}{E}, \qquad p_{(1)} \le \dots \le p_{(E)}.
\end{aligned}$$

- $G = 0$ for the uniform distribution; $G \to 1 - 1/E$ as mass collapses onto one
expert.
- Interpretation is the classical economic one: the area between the Lorenz curve of
the routing distribution and the line of perfect equality.
- Gini reacts most strongly to the *middle* of the distribution and complements both
entropy (which reacts smoothly everywhere) and $T_k$ (which reacts only to the
head).

A useful check is computing all three of $\tilde{H}$, $T_k$ and $G$ for the same layer.
If they tell the same story, the imbalance is unambiguous. If they disagree, the
distribution has interesting structure that a single number is hiding.

---

## 2. Comparing two distributions

Now fix a layer and look at two languages $d_1, d_2$ with distributions $p$ and $q$.
We want a scalar that captures *how differently* the router behaves under the two
inputs.

### 2.1 Kullback–Leibler divergence

$$\mathrm{KL}(p \,\|\, q) = \sum_i p_i \log \frac{p_i}{q_i}.$$

- Non-negative; $0$ iff $p = q$.
- Asymmetric and unbounded. $\mathrm{KL}(p \,\|\, q)$ blows up if any $q_i = 0$ where
$p_i > 0$; empirical zero counts are common, so naive KL is fragile.
- Useful interpretation: extra nats of code length needed to encode samples from
$p$ using a code optimal for $q$.

For empirical routing histograms, raw KL is rarely the right object on its own; it is
a building block for the symmetric quantities below, and it should be computed with
a small Laplace smoothing $\alpha$ added to the counts (the implementation does this).

### 2.2 Jensen–Shannon divergence

$$m = \tfrac{1}{2}(p + q), \qquad \mathrm{JS}(p, q) = \tfrac{1}{2}\mathrm{KL}(p \,\|\, m) + \tfrac{1}{2}\mathrm{KL}(q \,\|\, m).$$

- **Symmetric.**
- **Bounded:** $\mathrm{JS} \in [0, \log 2]$ in nats ($[0, 1]$ in bits).
- **Robust:** since $m$ is an average of $p$ and $q$, it is positive wherever either is,
so JS is well-defined for empirical histograms with zeros.
- $\sqrt{\mathrm{JS}}$ is a true metric on the simplex.

These properties make JS the workhorse for comparing empirical distributions, and the
default divergence used throughout this analysis.

**In the plot.** A pairwise JS matrix at a fixed layer is a *language-similarity matrix*
seen from the router's point of view. Hierarchical clustering or a 2-D MDS projection of
this matrix typically recovers known linguistic structure (script families, related
languages cluster) when the layer is doing language-specific work, and looks unstructured
on layers that route language-agnostically.

### 2.3 Other distances worth knowing

- **Total variation:** $\mathrm{TV}(p, q) = \tfrac{1}{2}\sum_i |p_i - q_i|$. Clean
interpretation: the fraction of routing mass that would need to be moved to turn
$p$ into $q$.
- **Hellinger:** $H(p, q) = \tfrac{1}{\sqrt{2}}\lVert \sqrt{p} - \sqrt{q} \rVert_2$. Downweights both
the busiest and the rarest experts.
- **Earth Mover's / Wasserstein:** *not* useful here, because expert IDs have no
meaningful ordering.

---

## 3. Comparing many distributions: generalized JS

For $L$ language distributions $p^{(1)}, \dots, p^{(L)}$ with non-negative weights
$w_1, \dots, w_L$ summing to 1, the *generalized* Jensen–Shannon divergence is

$$\mathrm{JS}_w\left(p^{(1)}, \dots, p^{(L)}\right) = H\left(\sum_{\ell=1}^{L} w_\ell p^{(\ell)}\right) - \sum_{\ell=1}^{L} w_\ell H\left(p^{(\ell)}\right).$$

This is the *information radius* of the family $p^{(\ell)}$. The intuition is the same as
for two-way JS:

- It equals 0 iff all distributions are identical.
- It is bounded by $\log L$ (achieved when the supports are disjoint).
- It admits the decomposition $\mathrm{JS}_w = \sum_{\ell=1}^{L} w_\ell \mathrm{KL}\!\left(p^{(\ell)} \,\|\, \bar{p}\right)$
with $\bar{p} = \sum_\ell w_\ell p^{(\ell)}$. Each language's KL to the pooled
distribution is its *language-specificity contribution* at that layer.

**In the plot.** One scalar per layer summarizes how language-sensitive that layer is,
collapsing the $L \times L$ JS matrix into a single curve over depth. Peaks identify
the layers worth zooming into with the pairwise JS matrix.

### Connection to ANOVA-style decomposition

$$H(\bar{p}) = \underbrace{\sum_{\ell=1}^{L} w_\ell H(p^{(\ell)})}_{\text{within-language uncertainty}} + \underbrace{\mathrm{JS}_w\!\left(p^{(1)}, \dots, p^{(L)}\right)}_{\text{between-language divergence}}.$$

This is the discrete-distribution analog of the within / between sum-of-squares split in
ANOVA. The total entropy of pooled routing decomposes into an average of within-language
entropies and a between-language information radius — exactly the question we want to
ask.

---

## 4. Localizing differences to specific experts

A single number, even a good one, hides *where* a divergence comes from. The two-way JS
admits a per-coordinate decomposition, where $m = \tfrac{1}{2}(p + q)$:

$$\mathrm{JS}(p, q) = \sum_{i=1}^E c_i(p,q), \qquad c_i(p,q) = \tfrac{1}{2}\left[p_i \log \frac{p_i}{m_i} + q_i \log \frac{q_i}{m_i}\right] \ge 0.$$

Each $c_i(p, q)$ is non-negative and they sum exactly to $\mathrm{JS}(p, q)$, so the
share $c_i / \mathrm{JS}$ is well-defined and interpretable as the *fraction of the
divergence between $p$ and $q$ explained by expert $i$*.

**In the plot.** Sorting experts by $c_i$ and showing the top $N$ as paired bars
($p_i$ for language A and $q_i$ for language B) answers two questions at once:

- *Which experts* are responsible for the divergence at this layer?
- *Is the divergence concentrated or diffuse?* If the top 12 experts already explain
  > 80% of $\mathrm{JS}$, the disagreement is local and you can describe it by listing
  > a few experts. If they explain only 30%, the disagreement is broad.

This is also the right place to look for *language-specific specialists*: experts whose
prob mass is high under one language and near-zero under another contribute strongly to
$c_i$ and stand out in the bar plot.

---

## 5. Practical advice for interpretation

Because the histograms are estimated from finite samples, a few cautions matter.

- **Counts vs. prob mass.** All metrics are functionals of the normalized distribution
$p$. Two layers with very different absolute traffic but the same shape will look
identical to entropy, JS, etc. — which is usually what you want, but worth flagging
when you compare layers with very different `total_routed_selections`.
- **Zero counts.** Empirical zeros are noise as much as they are signal. Add a small
Laplace smoothing $\alpha$ (e.g. $10^{-6}$ of the mean count) before computing KL
or symmetric KL. JS does not need this in principle, but smoothing also makes
$c_i$ more stable for low-traffic experts.
- **Per-layer baselines.** The same value of $\matrm{JS}$ means different things
for layers with very different baseline imbalance. A useful normalization is
$\mathrm{JS}(p, q) / \mathrm{JS}_w(p^{(\ell)})$: how much a *pair* of
languages contributes to the all-language divergence at that layer.
- **Multiple comparisons.** With $L$ languages there are $\binom{L}{2}$ language pairs
and dozens of layers; if you want to call a specific (pair, layer) divergence
significant, control the family-wise error or use a permutation test that shuffles
the language label and recomputes JS to obtain a null distribution.
- **Bootstrapping.** If you still have access to the per-batch counts, a bootstrap over
batches gives confidence intervals on $\tilde{H}$ and $\mathrm{JS}$ per layer and
is by far the cleanest way to check that observed differences are not sampling noise.

---

## 6. Reading the four plots together

The dashboard built in the notebook is structured to answer a chain of questions, each
one zooming in on what the previous answered:

1. **Section 1 — imbalance per layer per language.** Where in the network is routing
  specialized at all? Are some languages systematically more concentrated than others?
2. **Section 2 — generalized JS per layer.** *Given* that some layers are specialized,
  which of them route differently for different languages? One scalar per layer.
3. **Section 3 — pairwise JS at a chosen layer.** *Given* a language-divergent layer,
  which languages are similar to which? Look for block structure that matches known
   linguistic groupings.
4. **Section 4 — pointwise JS contributions.** *Given* a divergent pair at a divergent
  layer, which specific experts implement the difference, and is the difference
   concentrated on a few experts or smeared across many?

Used together they characterize the router's behavior along three axes: depth
(*where in the model*), language structure (*who differs from whom*), and expert
identity (*which subnetwork is responsible*).
