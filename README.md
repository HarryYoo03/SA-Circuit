# SA-Circuit

The official repository for the S.A Circuit — Compassion Constant (Cₐ) and Restorative Determinism framework.

**Author ORCID**  
https://orcid.org/0009-0009-4937-1025

**Journal Article (OSF)**  
https://doi.org/10.17605/OSF.IO/WSN6E

**Software Release (Zenodo DOI)**  
https://doi.org/10.5281/zenodo.17718241

## Equation Validation Guide  
**The S.A Circuit — Compassion Convergence Equation (Yoo, 2025)**  
This section provides a reproducible method for validating the Compassion Constant (Cₐ)  
and the ΨR(t) → ΨR(t+1) convergence dynamics.

---

### 1.Core Equation  
\[
\Psi_R(t+1) = \Psi_R(t) 
+ \frac{\alpha_t\left( \frac{dR}{dT}\big|_A - C_{\text{crit}} \right)}
{1 - \Psi_R(t)} \, \varepsilon_t
\]

Where:
- **Cₐ = dR/dT |ₐ**  
- **ΨR(t)** = restorative convergence level  
- **αₜ** = adaptive responsiveness  
- **εₜ** = perturbation factor  
- **C_crit** = minimum compassion threshold for stability  

---

### 2.Experimental Validation Protocol (v1.0)
This protocol reproduces the dynamics described in the OSF publication.

#### **Step 1 — Initialize parameters**
```python
Psi = 0.32          # initial restorative state
alpha = 0.12        # responsiveness
C_a = 0.41          # Compassion Constant (empirical)
C_crit = 0.33       # critical threshold
epsilon = 0.08      # perturbation factor


def update(Psi, alpha, C_a, C_crit, eps):
    return Psi + (alpha * (C_a - C_crit) / (1 - Psi)) * eps


trajectory = []

for t in range(60):
    Psi = update(Psi, alpha, C_a, C_crit, epsilon)
    trajectory.append(Psi)


for i, v in enumerate(trajectory[0:6]):
    print(f"t={i*10:>2}   {v:.3f}")

t=10  0.358
t=20  0.402
t=30  0.447
t=40  0.489
t=50  0.526

def update(Psi, alpha, C_a, C_crit, eps):
    return Psi + (alpha * (C_a - C_crit) / (1 - Psi)) * eps

### 2. Ca Measurement Toolkit (Python)

The code below estimates the Compassion Constant \(C_a = \frac{dR}{dT}\big\vert_A\)  
from real data and checks how stable it is across episodes.

**Expected CSV format**

- `t` : time index
- `R` : restorative / recovery level
- `T` : intervention / empathy input level
- `A` : activation flag (1 = within compassion episode, 0 = baseline)
- `episode` : episode id (participant / session / condition, etc.)

```python
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. Core estimator: C_a per episode
# --------------------------------------------------

def estimate_Ca_for_episode(df_ep, min_points=3):
    """
    Estimate C_a = dR/dT | A=1 for a single episode.
    Uses finite differences inside active segments (A == 1).
    """
    # Use only the active section
    active = df_ep[df_ep["A"] == 1].copy()
    active = active.sort_values("t")

    if len(active) < min_points:
        return np.nan

    dR = np.diff(active["R"].values)
    dT = np.diff(active["T"].values)

    # Prevent dividing by 0
    valid = dT != 0
    if valid.sum() == 0:
        return np.nan

    dR_dT = dR[valid] / dT[valid]

    # Use the median value to reduce noise (you can change it to average)
    Ca_hat = np.median(dR_dT)
    return Ca_hat


def estimate_Ca_global(df):
    """
    Estimate C_a for each episode and aggregate.
    Returns:
        - per_episode: DataFrame with C_a per episode
        - summary: dict with global statistics (mean, std, CV, n_episodes)
    """
    per_episode = (
        df.groupby("episode")
          .apply(estimate_Ca_for_episode)
          .rename("C_a_hat")
          .reset_index()
    )

    valid = per_episode["C_a_hat"].dropna().values
    if len(valid) == 0:
        summary = {
            "C_a_mean": np.nan,
            "C_a_std": np.nan,
            "C_a_cv": np.nan,  # coefficient of variation
            "n_episodes": 0,
        }
        return per_episode, summary

    mean = valid.mean()
    std = valid.std(ddof=1)
    cv = std / mean if mean != 0 else np.nan

    summary = {
        "C_a_mean": mean,
        "C_a_std": std,
        "C_a_cv": cv,
        "n_episodes": len(valid),
    }
    return per_episode, summary


# --------------------------------------------------
# 2. Simple stability check (constancy over time)
# --------------------------------------------------

def Ca_time_profile(df, window_size=10):
    """
    Optional: compute a sliding-window estimate of C_a over time
    to see whether it drifts.
    """
    df = df.sort_values("t").copy()
    times = []
    Ca_vals = []

    for start in range(0, len(df) - window_size + 1):
        win = df.iloc[start:start + window_size]
        Ca_win = estimate_Ca_for_episode(win, min_points=3)
        times.append(win["t"].mean())
        Ca_vals.append(Ca_win)

    return pd.DataFrame({"t_center": times, "C_a_hat": Ca_vals})


# --------------------------------------------------
# 3. Usage example
# --------------------------------------------------

if __name__ == "__main__":
    # 1) Load your CSV file
    df = pd.read_csv("your_experiment_data.csv")

    # 2) Estimate C_a per episode and global summary
    per_ep, summary = estimate_Ca_global(df)
    print("Per-episode C_a estimates:")
    print(per_ep)

    print("\nGlobal summary:")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # 3) Optional: check drift over time within the whole dataset
    profile = Ca_time_profile(df, window_size=12)
    print("\nTime-profile of C_a (first few rows):")
    print(profile.head())

For full experimental datasets and sample CSV structures,  
please refer to the official OSF Appendix files included in the Doi

Yoo, Harry. (2025). The S.A Circuit — Unified Law of Compassion and Integrative Science.