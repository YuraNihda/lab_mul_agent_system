# ядро: gini, Forager, SocietyModel, run() + базовий 

# lab1_society.py — «Суспільство агентів»: збирачі ресурсів на сітці (Mesa 3.x)
import os

import numpy as np
import matplotlib.pyplot as plt
from mesa import Model, DataCollector
from mesa.discrete_space import CellAgent, OrthogonalMooreGrid

# Папка для результатів (створюється поруч із файлом, незалежно від того,
# з якої директорії запущено скрипт)
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)


def gini(values):
    """Коефіцієнт Джині для списку запасів (0 — рівність, 1 — максимальна нерівність)."""
    x = np.sort(np.array(values, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return 1 - 2 * np.sum(cum) / (n * cum[-1]) + 1 / n


class Forager(CellAgent):
    """Агент-збирач: має запас енергії, локальне правило руху, витрачає метаболізм."""

    def __init__(self, model, cell, strategy, vision=1, metabolism=2):
        super().__init__(model)
        self.cell = cell                # розміщення у клітинці сітки
        self.strategy = strategy        # "greedy" | "random" | "social"
        self.vision = vision            # радіус огляду
        self.metabolism = metabolism    # витрати енергії за крок
        self.energy = self.random.randint(5, 15)

    def step(self):
        # 1. Спостереження: сусідні клітинки в радіусі vision (разом із поточною)
        cells = list(self.cell.get_neighborhood(radius=self.vision, include_center=True))

        # 2. Вибір дії за локальним правилом
        if self.strategy == "greedy":
            target = max(cells, key=lambda c: self.model.resource[c.coordinate])
        elif self.strategy == "social":
            # уникає переповнених клітинок: ресурс мінус штраф за кожного, хто вже там
            target = max(
                cells,
                key=lambda c: self.model.resource[c.coordinate] - 3 * len(c.agents),
            )
        else:  # random
            target = self.random.choice(cells)
        self.cell = target

        # 3. Дія: збір ресурсу (ділиться між усіма агентами в клітинці) і метаболізм
        share = self.model.resource[target.coordinate] / len(target.agents)
        self.energy += share
        self.model.resource[target.coordinate] -= share
        self.energy -= self.metabolism

        # 4. Смерть при вичерпанні енергії
        if self.energy <= 0:
            self.remove()


class SocietyModel(Model):
    """Середовище: тор WxH, ресурс регенерує зі швидкістю regrow до максимуму cap."""

    def __init__(self, n_agents=80, width=20, height=20, strategy="greedy",
                 regrow=0.3, cap=6.0, vision=1, metabolism=2, seed=None):
        super().__init__(rng=seed)
        self.grid = OrthogonalMooreGrid((width, height), torus=True, random=self.random)
        self.resource = self.rng.uniform(0, cap, size=(width, height))
        self.regrow, self.cap = regrow, cap

        for _ in range(n_agents):
            strat = (strategy if strategy != "mixed"
                     else self.random.choice(["greedy", "random", "social"]))
            Forager(self, self.grid.select_random_empty_cell(), strat, vision, metabolism)

        self.datacollector = DataCollector(
            model_reporters={
                "alive": lambda m: len(m.agents),
                "resource_total": lambda m: float(m.resource.sum()),
                "gini": lambda m: gini([a.energy for a in m.agents]),
                "mean_energy": lambda m: float(np.mean([a.energy for a in m.agents]))
                if len(m.agents) else 0.0,
            },
            agent_reporters={"energy": "energy", "strategy": "strategy"},
        )
        self.datacollector.collect(self)

    def step(self):
        self.agents.shuffle_do("step")                                   # усі агенти діють у випадковому порядку
        self.resource = np.minimum(self.resource + self.regrow, self.cap)  # регенерація ресурсу
        self.datacollector.collect(self)


def run(strategy, steps=200, seed=42, **kw):
    """Один прогін моделі. Додаткові параметри (regrow, n_agents...) передаються в SocietyModel."""
    model = SocietyModel(strategy=strategy, seed=seed, **kw)
    for _ in range(steps):
        model.step()
    return model


if __name__ == "__main__":
    # Базовий запуск: три чисті стратегії, seed=42
    results = {s: run(s) for s in ["greedy", "random", "social"]}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for s, m in results.items():
        df = m.datacollector.get_model_vars_dataframe()
        axes[0].plot(df["alive"], label=s)
        axes[1].plot(df["resource_total"], label=s)
        axes[2].plot(df["gini"], label=s)
        print(f"{s:8s} alive={df['alive'].iloc[-1]:3d} "
              f"resource={df['resource_total'].iloc[-1]:7.1f} gini={df['gini'].iloc[-1]:.3f}")
    for ax, title in zip(axes, ["Живі агенти", "Сумарний ресурс", "Джині (енергія)"]):
        ax.set_title(title)
        ax.set_xlabel("крок")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "lab1_dynamics.png"), dpi=120)
    plt.close()

    # Теплова карта ресурсу з позиціями агентів (greedy, кінець прогону)
    m = results["greedy"]
    plt.figure(figsize=(5, 5))
    plt.imshow(m.resource.T, origin="lower", cmap="YlGn")
    plt.scatter([a.cell.coordinate[0] for a in m.agents],
                [a.cell.coordinate[1] for a in m.agents], c="red", s=12)
    plt.title("Ресурс і агенти (greedy, крок 200)")
    plt.savefig(os.path.join(RESULTS, "lab1_map.png"), dpi=120)
    plt.close()