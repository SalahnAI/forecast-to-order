# Mathematical formalization

## 1. The newsvendor decision

For one product, one day, order quantity $Q$, random demand $D$ with CDF $F$:

- **Underage cost** $C_u$ — margin forgone per unit of unmet demand: $C_u = \text{price} - \text{cost}$
- **Overage cost** $C_o$ — cash lost per unit ordered but not sold: $C_o = \text{cost} - \text{salvage}$

Expected cost of ordering $Q$:

$$\mathcal{L}(Q) = C_u\,\mathbb{E}[(D-Q)^+] + C_o\,\mathbb{E}[(Q-D)^+]$$

Setting the derivative to zero ($\frac{d}{dQ}\mathbb{E}[(Q-D)^+] = F(Q)$, $\frac{d}{dQ}\mathbb{E}[(D-Q)^+] = F(Q) - 1$):

$$-C_u\,(1 - F(Q^*)) + C_o\,F(Q^*) = 0
\quad\Longrightarrow\quad
F(Q^*) = \frac{C_u}{C_u + C_o} \equiv q^*$$

**The optimal order is the $q^*$-quantile of the demand distribution.** With zero salvage, $q^* = \text{margin}/\text{price}$: a 50%-margin product is ordered at the median, a low-margin perishable well below it. The mean is optimal only in the knife-edge case $C_u = C_o$ — this is the entire case for forecasting distributions rather than points.

## 2. Why pinball loss is the right training objective

The pinball (quantile) loss at level $q$,

$$\ell_q(y, \hat{y}) = \max\big(q\,(y-\hat{y}),\,(q-1)\,(y-\hat{y})\big),$$

is minimized in expectation by the true conditional quantile $F^{-1}(q)$. It is, up to scaling, *the newsvendor cost itself* with $C_u \propto q$ and $C_o \propto 1-q$: training with pinball loss at the critical ratio directly minimizes the expected cost of the downstream decision. Forecast metric and business objective coincide.

## 3. Pack-size constraint

Orders come in packs of $m$ units. The policy is order-up-to with integer rounding:

$$Q = m \left\lceil \frac{(F^{-1}(q) - I)^+}{m} \right\rceil$$

where $I$ is the current inventory position (on-shelf + in-transit). Rounding *up* is the natural default when $C_u > C_o$ effective; a refinement compares the expected cost of the floor and ceiling pack counts per line.

## 4. Shelf life makes the single-period model conservative

The classic derivation assumes every unsold unit is lost at end of day. With shelf life $L > 1$, an unsold unit is wasted only if it fails to sell for $L$ consecutive days, so the *effective* overage cost is

$$C_o^{\text{eff}} = C_o \cdot \Pr(\text{unit expires unsold}) < C_o
\quad\Longrightarrow\quad
q^{\text{eff}} = \frac{C_u}{C_u + C_o^{\text{eff}}} > q^*.$$

$\Pr(\text{unit expires unsold})$ depends on the whole demand trajectory and the FIFO consumption order, which is why this repo measures the gap by **simulation** instead of closing it in formula: the empirical profit-maximizing quantile ($\approx 0.6$) sits above the average critical ratio ($\approx 0.4$), exactly as this argument predicts.

## 5. Evaluation metrics

- **Pinball loss** averaged over quantile levels — proper scoring for the quantile set.
- **Coverage** of the nominal 80% interval $[\hat{F}^{-1}(0.1), \hat{F}^{-1}(0.9)]$ — calibration check; ordering at quantile $q$ is only meaningful if predicted quantiles are calibrated.
- **wMAPE** $= \sum|y-\hat{y}| \,/\, \sum y$ — scale-free point accuracy, robust to zero-demand days (unlike MAPE).
- **MedMAPE** — median across series of per-series wMAPE; the long tail of erratic slow movers cannot dominate it.
