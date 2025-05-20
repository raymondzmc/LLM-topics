#!/usr/bin/env python3
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import statsmodels.api as sm
import statsmodels.formula.api as smf

# -------------------------------------------------------------------
# 0.  Parameters & paths
# -------------------------------------------------------------------
PROMPT_DIRS = [
    f"results/stackoverflow/Llama-3.2-1B-Instruct_vocab_2000_last_variant_{k}"
    for k in (1, 2, 3, 4)
]
BASELINES = {
    "lda":      "LDA",
    "prodlda":  "ProdLDA",
    "zeroshot": "Zeroshot",
    # "bertopic": "BERTopic",
}
METRIC_JSON_KEY = "llm_rating"        # key name inside evaluation_results.json
NUM_TOPICS = 100
FIG_OUT = f"llm_and_irbo_by_model_K{NUM_TOPICS}.png"
TXT_OUT = f"prompt_variance_K{NUM_TOPICS}.txt"


# -------------------------------------------------------------------
# 1.  Collect all runs  → rows = model · prompt · seed · llm_score · irbo
# -------------------------------------------------------------------
rows_plot   = []     # for box-plots   (pooled Llama+baselines)
rows_llama  = []     # for ANOVA only (Llama runs with prompt label)

for p_idx, root in enumerate(PROMPT_DIRS, start=1):
    kl_dir = os.path.join(root, f"{NUM_TOPICS}_KL")
    for seed_dir in sorted(d for d in os.listdir(kl_dir)
                           if os.path.isdir(os.path.join(kl_dir, d))):
        with open(os.path.join(kl_dir, seed_dir, "evaluation_results.json")) as f:
            res = json.load(f)
        rows_plot.append(
            dict(model="Llama-1B\n(5 prompts)", seed=seed_dir,
                 llm_score=res[METRIC_JSON_KEY], irbo=res["inverted_rbo"])
        )
        rows_llama.append(
            dict(model="Llama-1B",
                 prompt=f"p{p_idx}",
                 seed=seed_dir,
                 llm_score=res[METRIC_JSON_KEY],
                 irbo=res["inverted_rbo"])
        )

for tag, nice in BASELINES.items():
    kl_dir = os.path.join("results", "stackoverflow", f"{tag}_K{NUM_TOPICS}")
    for seed_dir in sorted(d for d in os.listdir(kl_dir)
                           if os.path.isdir(os.path.join(kl_dir, d))):
        with open(os.path.join(kl_dir, seed_dir, "evaluation_results.json")) as f:
            res = json.load(f)
        rows_plot.append(
            dict(model=nice, seed=seed_dir,
                 llm_score=res[METRIC_JSON_KEY], irbo=res["inverted_rbo"])
        )

df_plot  = pd.DataFrame(rows_plot)
df_llama = pd.DataFrame(rows_llama)        # 20 rows = 4 prompts × 5 seeds

# -------------------------------------------------------------------
# 2.  Dual-axis box-plot (unchanged)
# -------------------------------------------------------------------
order = (df_plot.groupby("model")["llm_score"].mean()
               .sort_values(ascending=True).index.tolist())
df_plot["model"] = pd.Categorical(df_plot["model"], categories=order, ordered=True)
df_plot = df_plot.sort_values("model")

# Adjust figure size for two-column paper format
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

# Increase box width to fill more space
w  = .4  # wider boxes (was .25)
xl = np.arange(len(order)) - w/2
xr = np.arange(len(order)) + w/2
llm_vals  = [df_plot[df_plot.model == m]["llm_score"].values for m in order]
irbo_vals = [df_plot[df_plot.model == m]["irbo"].values      for m in order]

# Create boxplots with wider width
b1 = ax1.boxplot(llm_vals, positions=xl, patch_artist=True,
                widths=w, showfliers=False, whis=[0, 100])
for p in b1["boxes"]: p.set_facecolor("#3274A1")
b2 = ax2.boxplot(irbo_vals, positions=xr, patch_artist=True,
                widths=w, showfliers=False, whis=[0, 100])
for p in b2["boxes"]: p.set_facecolor("#E1812C")

# Increase font sizes for better readability
FONT_SIZE = 14
plt.rcParams.update({'font.size': FONT_SIZE})
ax1.set_xticks(np.arange(len(order)))
ax1.set_xticklabels(order, fontsize=FONT_SIZE-1, rotation=0)
ax1.set_ylabel("LLM score", fontsize=FONT_SIZE+1)
ax2.set_ylabel("I-RBO", fontsize=FONT_SIZE+1)
ax1.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
ax2.tick_params(axis='both', which='major', labelsize=FONT_SIZE)

# Adjust legend for better visibility
ax1.legend(handles=[
    Patch(facecolor="#3274A1", edgecolor="black", label="LLM score"),
    Patch(facecolor="#E1812C", edgecolor="black", label="I-RBO")
], loc="center left", fontsize=FONT_SIZE)

# Ensure proper layout and higher resolution
fig.tight_layout()
fig.savefig(FIG_OUT, dpi=600, bbox_inches='tight')

# -------------------------------------------------------------------
# 3.  Prompt-effect ANOVA  (run for both metrics)
# -------------------------------------------------------------------
def anova_variance(df, score_col):
    """one-way (prompt) fixed-effect ANOVA + classical RE variance parts"""
    aov = sm.stats.anova_lm(
            smf.ols(f"{score_col} ~ C(prompt)", data=df).fit(), typ=2)
    MS_prompt = aov.loc["C(prompt)", "sum_sq"] / aov.loc["C(prompt)", "df"]
    MS_error  = aov.loc["Residual",  "sum_sq"] / aov.loc["Residual",  "df"]
    n_seeds   = df["seed"].nunique()        # 5
    var_prompt = max((MS_prompt - MS_error) / n_seeds, 0.0)
    var_seed   = MS_error
    var_total  = var_prompt + var_seed
    share = (100 * var_prompt / var_total) if var_total else 0.0
    return aov, var_prompt, var_seed, share

print(f"Num topics: {NUM_TOPICS}")
report_lines = []
for col, label in [("llm_score", "LLM score"),
                   ("irbo",      "inverted RBO")]:
    aov, v_prompt, v_seed, share = anova_variance(df_llama, col)
    report_lines.append(f"\n=== {label} ===\n{aov}\n"
                        f"σ²_prompt = {v_prompt:.6e}  "
                        f"({share:4.1f} % of total)\n"
                        f"σ²_seed   = {v_seed:.6e}\n")

with open(TXT_OUT, "w") as f:
    f.write("\n".join(report_lines))
print("\n".join(report_lines))
print(f"✓ Saved: {FIG_OUT}   {TXT_OUT}")
