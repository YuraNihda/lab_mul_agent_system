# експерименти 1-3: цикли по seed-ах, таблиці, графіки

# experiments.py — експерименти 1–3 до лабораторної №1
# Запуск (з папки lab1, з активним venv):  python experiments.py
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lab1_society import run, RESULTS

SEEDS = range(10)          # seed 0..9 (у звіті вказати!)
STEPS = 200                # кількість кроків прогону
TAIL = 50                  # стаціонарна ділянка: останні 50 кроків
METRICS = ["alive", "mean_energy", "gini", "resource_total"]
TITLES = {"alive": "Живі агенти", "mean_energy": "Середня енергія",
          "gini": "Джині (енергія)", "resource_total": "Сумарний ресурс"}
PURE = ["greedy", "random", "social"]


def stationary(model, tail=TAIL):
    """Середнє кожної метрики за останні `tail` кроків прогону."""
    df = model.datacollector.get_model_vars_dataframe()
    return df.iloc[-tail:].mean()


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, name), dpi=120)
    plt.close()


# ---------------------------------------------------------------- Експеримент 1
def experiment1():
    """Порівняння стратегій (greedy / random / social / mixed) на 10 seed-ах."""
    strategies = PURE + ["mixed"]
    rows, series = [], {s: [] for s in strategies}

    for s in strategies:
        for seed in SEEDS:
            m = run(s, steps=STEPS, seed=seed)
            row = stationary(m).to_dict()
            row.update(strategy=s, seed=seed)
            rows.append(row)
            series[s].append(m.datacollector.get_model_vars_dataframe())

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(RESULTS, "exp1_raw.csv"), index=False)

    summary = raw.groupby("strategy")[METRICS].agg(["mean", "std"]).loc[strategies]
    summary.to_csv(os.path.join(RESULTS, "exp1_strategies.csv"))
    print("\n=== Експеримент 1: середнє / std за останні 50 кроків, 10 seed-ів ===")
    print(summary.round(2).to_string())

    # Стовпчики: стратегія → метрика (середнє ± std)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, metric in zip(axes, METRICS):
        means = [raw[raw.strategy == s][metric].mean() for s in strategies]
        stds = [raw[raw.strategy == s][metric].std() for s in strategies]
        ax.bar(strategies, means, yerr=stds, capsize=4,
               color=["#4c72b0", "#dd8452", "#55a868", "#8172b3"])
        ax.set_title(TITLES[metric])
    save("exp1_strategies.png")

    # Динаміка: середнє по seed-ах зі смугою ±std
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ["alive", "resource_total", "gini"]):
        for s in strategies:
            data = np.array([df[metric].to_numpy() for df in series[s]])
            mean, std = data.mean(axis=0), data.std(axis=0)
            ax.plot(mean, label=s)
            ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.15)
        ax.set_title(TITLES[metric])
        ax.set_xlabel("крок")
        ax.legend()
    save("exp1_dynamics.png")


# ---------------------------------------------------------------- Експеримент 2
def experiment2(regrows=(0.1, 0.2, 0.3, 0.5, 0.8), n_agents=80, metabolism=2,
                width=20, height=20):
    """Вплив швидкості регенерації на виживання (стратегія greedy)."""
    rows = []
    for rg in regrows:
        for seed in SEEDS:
            m = run("greedy", steps=STEPS, seed=seed, regrow=rg,
                    n_agents=n_agents, metabolism=metabolism,
                    width=width, height=height)
            row = stationary(m).to_dict()
            row.update(regrow=rg, seed=seed)
            rows.append(row)

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(RESULTS, "exp2_raw.csv"), index=False)
    summary = raw.groupby("regrow")[["alive", "resource_total"]].agg(["mean", "std"])

    # Баланс: споживання (metabolism * n_agents) проти відновлення (regrow * W * H)
    consumption = metabolism * n_agents
    summary["regen_total"] = [rg * width * height for rg in summary.index]
    summary["consumption"] = consumption
    summary.to_csv(os.path.join(RESULTS, "exp2_regrow.csv"))
    print("\n=== Експеримент 2: вплив regrow (greedy) ===")
    print(summary.round(2).to_string())
    print(f"Теоретичний поріг: regrow* = metabolism*n_agents/(W*H) = "
          f"{consumption / (width * height):.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, ylabel in zip(axes, ["alive", "resource_total"],
                                  ["Виживші агенти", "Сумарний ресурс"]):
        ax.errorbar(summary.index, summary[(metric, "mean")],
                    yerr=summary[(metric, "std")], marker="o", capsize=4)
        ax.set_xlabel("regrow")
        ax.set_ylabel(ylabel)
        ax.set_title(f"regrow → {ylabel.lower()}")
    axes[0].axhline(n_agents, ls="--", c="gray", label="початкова кількість")
    axes[0].axvline(consumption / (width * height), ls=":", c="red",
                    label="поріг за балансом")
    axes[0].legend()
    save("exp2_regrow.png")


# ---------------------------------------------------------------- Експеримент 3
def experiment3():
    """Змішана популяція проти чистих стратегій; перевірка ефекту різноманітності."""
    rows, alive = [], {s: [] for s in PURE + ["mixed"]}

    for seed in SEEDS:
        m = run("mixed", steps=STEPS, seed=seed)
        adf = m.datacollector.get_agent_vars_dataframe().reset_index()
        start = Counter(adf[adf["Step"] == 0]["strategy"])      # склад на початку
        end = Counter(a.strategy for a in m.agents)             # склад наприкінці
        row = {"seed": seed, "alive_total": sum(end.values())}
        for s in PURE:
            row[f"start_{s}"] = start.get(s, 0)
            row[f"end_{s}"] = end.get(s, 0)
            row[f"survival_{s}"] = end.get(s, 0) / start[s] if start.get(s, 0) else np.nan
        rows.append(row)
        alive["mixed"].append(stationary(m)["alive"])

    for s in PURE:
        for seed in SEEDS:
            alive[s].append(stationary(run(s, steps=STEPS, seed=seed))["alive"])

    mixed = pd.DataFrame(rows)
    mixed.to_csv(os.path.join(RESULTS, "exp3_mixed.csv"), index=False)
    print("\n=== Експеримент 3: змішана популяція ===")
    print(mixed.round(2).to_string(index=False))
    print("\nСередня частка вцілілих у mixed за стратегіями:")
    print(mixed[[f"survival_{s}" for s in PURE]].mean().round(3).to_string())

    means = {s: np.mean(v) for s, v in alive.items()}
    stds = {s: np.std(v, ddof=1) for s, v in alive.items()}
    best = max(PURE, key=lambda s: means[s])
    print("\nЖиві агенти (середнє за останні 50 кроків): "
          + ", ".join(f"{s}={means[s]:.1f}±{stds[s]:.1f}" for s in alive))
    print(f"Найкраща чиста стратегія: {best}. "
          f"mixed − {best} = {means['mixed'] - means[best]:+.1f}")

    try:  # статистичний тест (потрібен scipy; необов'язково)
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(alive["mixed"], alive[best], alternative="two-sided")
        print(f"Манна–Вітні (mixed vs {best}): U={stat:.1f}, p={p:.4f}")
    except ImportError:
        print("scipy не встановлено — статистичний тест пропущено (pip install scipy)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    surv = mixed[[f"survival_{s}" for s in PURE]]
    axes[0].bar(PURE, surv.mean(), yerr=surv.std(), capsize=4,
                color=["#4c72b0", "#dd8452", "#55a868"])
    axes[0].set_title("Mixed: частка вцілілих за стратегіями")
    axes[0].set_ylabel("вижили / було на початку")
    axes[1].boxplot([alive[s] for s in alive])
    axes[1].set_xticklabels(list(alive))
    axes[1].set_title("Живі агенти: чисті стратегії vs mixed")
    save("exp3_mixed.png")


if __name__ == "__main__":
    experiment1()
    experiment2()
    experiment3()
    print(f"\nГотово. Файли збережено в {RESULTS}")