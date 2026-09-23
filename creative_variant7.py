# творче завдання (N = ваш варіант): нові агенти, метрика, візуалізаціяґ

# creative_variant7.py — Творче завдання, варіант 7: «Кіберзахист» (Mesa 3.x)
#
# Сценарій: мережа з 60 вузлів; 5 вузлів спочатку заражені. Заражений агент (Malware)
# щокроку з ймовірністю p_spread намагається заразити випадкового сусіда.
# 3 патрульні (Patrol) рухаються мережею за локальним правилом:
#   «іти до вузла з найбільшою підозрілою активністю» і лікують заражений вузол.
# Головна метрика — частка заражених вузлів.
import os

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mesa import Model, Agent, DataCollector

from lab1_society import RESULTS   # та сама папка results/


class Malware(Agent):
    """Заражений агент: живе у вузлі й намагається заразити випадкового сусіда."""

    def __init__(self, model, node):
        super().__init__(model)
        self.node = node

    def step(self):
        if self.node is None:          # агента вже видалив патруль у цьому кроці
            return
        if self.random.random() < self.model.p_spread:
            neighbour = self.random.choice(list(self.model.G.neighbors(self.node)))
            self.model.infect(neighbour)


class Patrol(Agent):
    """Патрульний: реактивний агент із локальним правилом руху по мережі."""

    def __init__(self, model, node, strategy="suspicion", vision=2):
        super().__init__(model)
        self.node = node
        self.strategy = strategy      # "suspicion" | "random"
        self.vision = vision          # радіус огляду в «стрибках» по мережі; None = уся мережа
        self.cleaned = 0

    def step(self):
        m = self.model
        # 1. Спостереження: вузли в радіусі vision (разом із поточним)
        if self.vision is None:
            candidates = list(m.G.nodes)
        else:
            candidates = list(nx.single_source_shortest_path_length(
                m.G, self.node, cutoff=self.vision))

        # 2. Вибір дії за локальним правилом
        if self.strategy == "suspicion":
            # до вузла з найбільшою підозрілою активністю (рівні значення — навмання)
            target = max(candidates, key=lambda n: (m.suspicion[n], self.random.random()))
        else:  # random — базова лінія
            target = self.random.choice(candidates)
        self.node = target

        # 3. Дія: лікування вузла, якщо він заражений
        if m.disinfect(target):
            self.cleaned += 1


class CyberModel(Model):
    """Мережа з n_nodes вузлів (граф Барабаші–Альберт: є «хаби», як в реальних мережах)."""

    def __init__(self, n_nodes=60, n_infected=5, n_patrols=3, patrol_strategy="suspicion",
                 vision=2, p_spread=0.5, decay=0.7, seed=None):
        super().__init__(rng=seed)
        self.n_nodes, self.p_spread, self.decay = n_nodes, p_spread, decay
        self.G = nx.barabasi_albert_graph(n_nodes, 2, seed=self.random.randrange(2**31))

        self.suspicion = np.zeros(n_nodes)              # «підозріла активність» у вузлах
        self.infected = np.zeros(n_nodes, dtype=bool)   # істинний стан (патрулі його НЕ бачать)
        self.malware_at = {}                            # вузол -> агент Malware
        self.cleaned_total = 0

        for node in self.random.sample(range(n_nodes), n_infected):
            self.infect(node)
        self.patrols = [Patrol(self, self.random.randrange(n_nodes), patrol_strategy, vision)
                        for _ in range(n_patrols)]
        self._update_suspicion()

        self.datacollector = DataCollector(model_reporters={
            "infected_frac": lambda m: len(m.malware_at) / m.n_nodes,    # головна метрика
            "infected": lambda m: len(m.malware_at),
            "cleaned_total": lambda m: m.cleaned_total,
            # наскільки сигнал корисний: яка частка 3 «найпідозріліших» вузлів справді заражена
            "top3_precision": lambda m: float(m.infected[np.argsort(m.suspicion)[-3:]].mean()),
        })
        self.datacollector.collect(self)

    def infect(self, node):
        if node not in self.malware_at:
            self.malware_at[node] = Malware(self, node)
            self.infected[node] = True

    def disinfect(self, node):
        """Лікує вузол; повертає True, якщо він був заражений."""
        if node not in self.malware_at:
            return False
        agent = self.malware_at.pop(node)
        agent.node = None
        agent.remove()
        self.infected[node] = False
        self.suspicion[node] = 0.0
        self.cleaned_total += 1
        return True

    def _update_suspicion(self):
        """Заражені вузли «шумлять» сильніше, але сигнал зашумлений і затухає."""
        signal = self.rng.uniform(0.3, 1.3, self.n_nodes)   # активність зараженого вузла
        noise = self.rng.uniform(0.0, 0.8, self.n_nodes)    # фонова активність здорового
        self.suspicion = self.decay * self.suspicion + np.where(self.infected, signal, noise)

    def step(self):
        self.agents.shuffle_do("step")     # і malware, і патрулі — у випадковому порядку
        self._update_suspicion()
        self.datacollector.collect(self)


def run(steps=100, seed=42, **kw):
    model = CyberModel(seed=seed, **kw)
    for _ in range(steps):
        model.step()
    return model


# ------------------------------------------------------------ візуалізація
def snapshot(model):
    return {"infected": model.infected.copy(), "suspicion": model.suspicion.copy(),
            "patrols": [p.node for p in model.patrols]}


def draw_state(ax, G, pos, st, title):
    nodes = list(G.nodes)
    colors = ["#d62728" if st["infected"][n] else "#2ca02c" for n in nodes]
    sizes = 40 + 35 * st["suspicion"][nodes]           # розмір вузла ~ підозріла активність
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=st["patrols"], node_color="none",
                           edgecolors="blue", node_size=320, linewidths=2.5, node_shape="s")
    ax.set_title(f"{title}  (заражено {int(st['infected'].sum())}/{len(nodes)})")
    ax.axis("off")


def experiment_and_plots(seeds=range(10), steps=100, tail=30):
    configs = {
        "без патрулів":            dict(n_patrols=0),
        "random-патруль":          dict(patrol_strategy="random", vision=2),
        "suspicion, vision=1":     dict(vision=1),
        "suspicion, vision=2":     dict(vision=2),
        "suspicion, вся мережа":   dict(vision=None),
    }

    rows, curves = [], {}
    for name, kw in configs.items():
        runs = []
        for seed in seeds:
            df = run(steps=steps, seed=seed, **kw).datacollector.get_model_vars_dataframe()
            runs.append(df["infected_frac"].to_numpy())
            rows.append({"конфігурація": name, "seed": seed,
                         "частка заражених (стац.)": df["infected_frac"].iloc[-tail:].mean(),
                         "пік": df["infected_frac"].max(),
                         "вилікувано": df["cleaned_total"].iloc[-1]})
        curves[name] = np.array(runs)

    raw = pd.DataFrame(rows)
    summary = raw.groupby("конфігурація", sort=False)[
        ["частка заражених (стац.)", "пік", "вилікувано"]
    ].agg(["mean", "std"])
    raw.to_csv(os.path.join(RESULTS, "creative_raw.csv"), index=False)
    summary.to_csv(os.path.join(RESULTS, "creative_summary.csv"))
    print(f"=== Кіберзахист: {len(list(seeds))} seed-ів, {steps} кроків, "
          f"стаціонарна ділянка — останні {tail} кроків ===")
    print(summary.round(3).to_string())

    # Графік 1: динаміка частки заражених (середнє ± std по seed-ах)
    plt.figure(figsize=(8, 4.5))
    for name, data in curves.items():
        mean, std = data.mean(axis=0), data.std(axis=0)
        plt.plot(mean, label=name)
        plt.fill_between(range(len(mean)), np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=0.15)
    plt.xlabel("крок")
    plt.ylabel("частка заражених вузлів")
    plt.ylim(0, 1.05)
    plt.title("Кіберзахист: вплив політики патрулювання")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "creative_infected.png"), dpi=120)
    plt.close()

    # Графік 2: стан мережі в різні моменти двох прогонів (suspicion, vision=2).
    # Один seed — успішне стримування, інший — епідемія: модель бістабільна.
    moments = [0, 10, 30, 80]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for row, (seed, label) in enumerate([(14, "стримано"), (6, "епідемія")]):
        model = CyberModel(seed=seed, vision=2)
        pos = nx.spring_layout(model.G, seed=1)
        shots = {}
        for t in range(max(moments) + 1):
            if t in moments:
                shots[t] = snapshot(model)
            model.step()
        for ax, t in zip(axes[row], moments):
            draw_state(ax, model.G, pos, shots[t], f"seed={seed} ({label}), крок {t}")
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", color="#d62728", label="заражений вузол"),
        Line2D([], [], marker="o", ls="", color="#2ca02c", label="здоровий вузол"),
        Line2D([], [], marker="s", ls="", mfc="none", mec="blue", mew=2, ms=10, label="патрульний"),
    ], loc="lower center", ncol=3)
    fig.suptitle("Мережа: розмір вузла ~ підозріла активність (suspicion, vision=2)")
    plt.tight_layout(rect=(0, 0.04, 1, 0.96))
    plt.savefig(os.path.join(RESULTS, "creative_network.png"), dpi=110)
    plt.close()


if __name__ == "__main__":
    experiment_and_plots()
    print(f"\nГотово. Файли збережено в {RESULTS}")