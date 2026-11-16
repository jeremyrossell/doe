import math
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


# ==============================================================================
# 1. CARGA DE CONFIGURACIÓN DESDE .env
# ==============================================================================
def cargar_env(ruta_env: str = ".env") -> None:
    """Carga variables de entorno desde un archivo .env sin dependencias externas."""
    path = Path(ruta_env)
    if not path.exists():
        path = Path(__file__).resolve().parent / ruta_env
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo {ruta_env} en {Path.cwd()}")

    for linea in path.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, valor = linea.split("=", 1)
        clave = clave.strip()
        valor = valor.strip().strip('"').strip("'")
        os.environ.setdefault(clave, valor)


# ==============================================================================
# 2. TSP Y RECOCIDO SIMULADO CON 2-OPT (TAREA 2, EJERCICIO 4)
# ==============================================================================
def generar_ciudades(num_ciudades: int, semilla: int) -> list[tuple[float, float]]:
    """Genera coordenadas 2D aleatorias en un plano de 100x100."""
    rng = random.Random(semilla)
    return [(rng.uniform(0.0, 100.0), rng.uniform(0.0, 100.0)) for _ in range(num_ciudades)]


def calcular_matriz_distancias(ciudades: list[tuple[float, float]]) -> list[list[float]]:
    """Precalcula la matriz simétrica de distancias euclidianas."""
    n = len(ciudades)
    return [[math.dist(ciudades[i], ciudades[j]) for j in range(n)] for i in range(n)]


def longitud_tour(tour: list[int], matriz_dist: list[list[float]]) -> float:
    """Calcula la distancia total del ciclo cerrado del TSP."""
    n = len(tour)
    return sum(matriz_dist[tour[i]][tour[(i + 1) % n]] for i in range(n))


def sucesor_2opt_aleatorio(
    tour: list[int], matriz_dist: list[list[float]], rng: random.Random
) -> tuple[list[int], float]:
    """Genera un vecino 2-opt aleatorio y calcula el cambio de costo (delta) en O(1)."""
    n = len(tour)
    i, k = sorted(rng.sample(range(n), 2))

    # Evitar inversión trivial del tour completo
    if i == 0 and k == n - 1:
        return list(tour), 0.0

    a, b = tour[i - 1], tour[i]
    c, d = tour[k], tour[(k + 1) % n]

    delta_costo = (matriz_dist[a][c] + matriz_dist[b][d]) - (
        matriz_dist[a][b] + matriz_dist[c][d]
    )
    nuevo_tour = tour[:i] + list(reversed(tour[i : k + 1])) + tour[k + 1 :]
    return nuevo_tour, delta_costo


def recocido_simulado_2opt(
    matriz_dist: list[list[float]],
    T_inicial: float,
    semilla: int,
    pasos_totales: int = 1_500,
    T_final: float = 0.01,
) -> tuple[float, float]:
    """
    Ejecuta Recocido Simulado con enfriamiento exponencial y vecindad 2-opt
    partiendo de una permutación inicial aleatoria.
    """
    rng = random.Random(semilla)
    n = len(matriz_dist)

    tour_actual = list(range(n))
    rng.shuffle(tour_actual)
    costo_actual = longitud_tour(tour_actual, matriz_dist)
    mejor_costo = costo_actual

    # Esquema de enfriamiento exponencial: T(k) = T_0 * alpha^k
    alpha = (T_final / T_inicial) ** (1.0 / pasos_totales)
    T = T_inicial

    t0 = time.perf_counter()
    for _ in range(pasos_totales):
        if T <= 1e-8:
            break

        siguiente_tour, delta = sucesor_2opt_aleatorio(tour_actual, matriz_dist, rng)

        if delta < 0.0:
            tour_actual = siguiente_tour
            costo_actual += delta
            if costo_actual < mejor_costo:
                mejor_costo = costo_actual
        else:
            if rng.random() < math.exp(-delta / T):
                tour_actual = siguiente_tour
                costo_actual += delta

        T *= alpha

    tiempo_ejecucion = time.perf_counter() - t0
    return mejor_costo, tiempo_ejecucion


# ==============================================================================
# 3. BOOTSTRAP PARA MEDIANA E INTERVALO DE CONFIANZA DEL 95%
# ==============================================================================
def bootstrap_mediana_ic95(
    datos: np.ndarray, n_boot: int = 10_000, semilla: int = 2173396
) -> tuple[float, float, float]:
    """Calcula la mediana observada y su IC del 95% mediante remuestreo bootstrap."""
    rng = np.random.default_rng(semilla)
    n = len(datos)
    remuestras = rng.choice(datos, size=(n_boot, n), replace=True)
    medianas_boot = np.median(remuestras, axis=1)

    mediana_obs = float(np.median(datos))
    ic_inf, ic_sup = np.percentile(medianas_boot, [2.5, 97.5])
    return mediana_obs, float(ic_inf), float(ic_sup)


# ==============================================================================
# 4. EJECUCIÓN DEL DISEÑO FACTORIAL COMPLETO (OPCIÓN A)
# ==============================================================================
def ejecutar_opcion_a() -> None:
    cargar_env(".env")

    # Crear output/ y output/option_A/ en el mismo directorio raíz que este script
    out_dir = Path(__file__).resolve().parent / "output" / "option_A"
    out_dir.mkdir(parents=True, exist_ok=True)

    matricula = int(os.environ["MATRICULA"])
    r_fijo = float(os.environ["R_FIXED"])
    tamanos = [int(x.strip()) for x in os.environ["TSP_SIZES"].split(",")]
    num_replicas = int(os.environ["REPLICAS"])

    niveles_temp = {
        "Bajo": float(os.environ["T_A_BAJO"]),
        "Medio": float(os.environ["T_A_MEDIO"]),
        "Alto": float(os.environ["T_A_ALTO"]),
    }

    print("=" * 78)
    print("OPCIÓN A: DISEÑO FACTORIAL COMPLETO (3 NIVELES DE TEMPERATURA × 3 TAMAÑOS)")
    print("=" * 78)
    print(f"Matrícula: {matricula} | r fijo = {r_fijo:.10f}")
    print(
        f"Bloques (n): {tamanos} | Réplicas por punto: {num_replicas} | "
        f"Total corridas: {len(niveles_temp) * len(tamanos) * num_replicas}"
    )
    for nombre, val in niveles_temp.items():
        print(f"  - Nivel {nombre:<6}: T_0 = {val:.4f}")
    print("-" * 78)

    # Generar UNA instancia fija por cada tamaño de problema n (bloque)
    instancias_dist = {}
    for n in tamanos:
        ciudades = generar_ciudades(n, semilla=matricula + n)
        instancias_dist[n] = calcular_matriz_distancias(ciudades)

    # Ejecutar las 90 corridas experimentales
    registros = []
    for n in tamanos:
        matriz_dist = instancias_dist[n]
        for nivel_nombre, T0 in niveles_temp.items():
            for rep in range(1, num_replicas + 1):
                semilla_corrida = matricula * 1000 + n * 100 + rep
                costo, tiempo_s = recocido_simulado_2opt(
                    matriz_dist=matriz_dist,
                    T_inicial=T0,
                    semilla=semilla_corrida,
                    pasos_totales=1_500,
                    T_final=0.01,
                )
                registros.append(
                    {
                        "tamano": n,
                        "temperatura": nivel_nombre,
                        "T0": T0,
                        "replica": rep,
                        "costo": costo,
                        "tiempo_s": tiempo_s,
                    }
                )

    df = pd.DataFrame(registros)
    orden_temp = ["Bajo", "Medio", "Alto"]
    df["temperatura"] = pd.Categorical(df["temperatura"], categories=orden_temp, ordered=True)
    df.to_csv(out_dir / "resultados_opcion_a.csv", index=False)

    # Resumen descriptivo por punto de diseño (3 x 3 = 9 celdas)
    print("\n--- RESUMEN DESCRIPTIVO POR PUNTO DE DISEÑO (n = 10 réplicas c/u) ---")
    resumen = (
        df.groupby(["tamano", "temperatura", "T0"], observed=True)["costo"]
        .agg(media="mean", std="std", mediana="median", min="min", max="max")
        .reset_index()
    )
    print(resumen.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --------------------------------------------------------------------------
    # ANOVA DE DOS FACTORES (α = 0.05)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TABLA ANOVA DE DOS FACTORES (α = 0.05 | IC 95%)")
    print("=" * 78)
    modelo = ols("costo ~ C(temperatura) * C(tamano)", data=df).fit()
    tabla_anova = sm.stats.anova_lm(modelo, typ=2)
    print(tabla_anova.to_string(float_format=lambda x: f"{x:.6f}"))

    p_temp = float(tabla_anova.loc["C(temperatura)", "PR(>F)"])
    p_tam = float(tabla_anova.loc["C(tamano)", "PR(>F)"])
    p_inter = float(tabla_anova.loc["C(temperatura):C(tamano)", "PR(>F)"])

    alpha_sig = 0.05
    sig_temp = p_temp < alpha_sig
    sig_tam = p_tam < alpha_sig
    sig_inter = p_inter < alpha_sig

    print("\n--- DIAGNÓSTICO DE SIGNIFICANCIA ESTADÍSTICA (α = 0.05) ---")
    print(
        f"  1. Efecto Principal Temperatura (T_0) : p = {p_temp:.6e} -> "
        f"{'SIGNIFICATIVO' if sig_temp else 'NO significativo'}"
    )
    print(
        f"  2. Efecto Principal Tamaño (n)        : p = {p_tam:.6e} -> "
        f"{'SIGNIFICATIVO' if sig_tam else 'NO significativo'}"
    )
    print(
        f"  3. Interacción Temperatura × Tamaño   : p = {p_inter:.6e} -> "
        f"{'SIGNIFICATIVA' if sig_inter else 'NO significativa'}"
    )

    # --------------------------------------------------------------------------
    # MEDIANA VÍA BOOTSTRAP (IC 95%) SOLO PARA EFECTOS/PUNTOS SIGNIFICATIVOS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("MEDIANA VÍA BOOTSTRAP (10,000 REMUESTREOS, IC 95%) — SOLO EFECTOS SIGNIFICATIVOS")
    print("=" * 78)

    if not (sig_temp or sig_tam or sig_inter):
        print("Ningún efecto resultó estadísticamente significativo (p >= 0.05).")
    else:
        if sig_inter:
            print("La interacción Temperatura × Tamaño es significativa (p < 0.05).")
            print("Se reporta la mediana bootstrap para los 9 puntos de diseño (Temperatura × Tamaño):\n")
            filas_boot = []
            for idx, ((n, temp_lvl), grupo) in enumerate(
                df.groupby(["tamano", "temperatura"], observed=True)
            ):
                med, inf, sup = bootstrap_mediana_ic95(
                    grupo["costo"].to_numpy(), n_boot=10_000, semilla=matricula + idx
                )
                filas_boot.append(
                    {
                        "tamano": n,
                        "temperatura": temp_lvl,
                        "T0": niveles_temp[temp_lvl],
                        "mediana_boot": med,
                        "IC95_inf": inf,
                        "IC95_sup": sup,
                    }
                )
            print(pd.DataFrame(filas_boot).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        if sig_temp and not sig_inter:
            print("Efecto principal Temperatura significativo (p < 0.05) — Mediana bootstrap por nivel de T_0:\n")
            filas_temp = []
            for idx, (temp_lvl, grupo) in enumerate(df.groupby("temperatura", observed=True)):
                med, inf, sup = bootstrap_mediana_ic95(
                    grupo["costo"].to_numpy(), n_boot=10_000, semilla=matricula + 100 + idx
                )
                filas_temp.append(
                    {
                        "temperatura": temp_lvl,
                        "T0": niveles_temp[temp_lvl],
                        "mediana_boot": med,
                        "IC95_inf": inf,
                        "IC95_sup": sup,
                    }
                )
            print(pd.DataFrame(filas_temp).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print()

        if sig_tam and not sig_inter:
            print("Efecto principal Tamaño significativo (p < 0.05) — Mediana bootstrap por bloque n:\n")
            filas_tam = []
            for idx, (n_val, grupo) in enumerate(df.groupby("tamano", observed=True)):
                med, inf, sup = bootstrap_mediana_ic95(
                    grupo["costo"].to_numpy(), n_boot=10_000, semilla=matricula + 200 + idx
                )
                filas_tam.append(
                    {
                        "tamano_n": n_val,
                        "mediana_boot": med,
                        "IC95_inf": inf,
                        "IC95_sup": sup,
                    }
                )
            print(pd.DataFrame(filas_tam).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print()

    # --------------------------------------------------------------------------
    # VISUALIZACIONES (MATPLOTLIB)
    # --------------------------------------------------------------------------
    generar_graficas_opcion_a(df, niveles_temp, out_dir)


def generar_graficas_opcion_a(
    df: pd.DataFrame, niveles_temp: dict[str, float], out_dir: Path
) -> None:
    """Genera la gráfica de efectos principales y la gráfica de interacción en out_dir."""
    media_global = df["costo"].mean()
    orden_temp = ["Bajo", "Medio", "Alto"]
    etiquetas_temp = [f"{k}\n(T₀={niveles_temp[k]:.2f})" for k in orden_temp]

    # 1. GRÁFICA DE EFECTOS PRINCIPALES
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)

    # Panel izquierdo: Efecto principal de Temperatura
    stats_temp = (
        df.groupby("temperatura", observed=True)["costo"]
        .agg(["mean", "count", "std"])
        .reindex(orden_temp)
    )
    ci95_temp = 1.96 * (stats_temp["std"] / np.sqrt(stats_temp["count"]))

    ax1.errorbar(
        range(len(orden_temp)),
        stats_temp["mean"],
        yerr=ci95_temp,
        fmt="-o",
        color="tab:blue",
        linewidth=2.2,
        markersize=8,
        capsize=5,
        label="Media ± IC 95%",
    )
    ax1.axhline(
        media_global,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label=f"Media global ({media_global:.2f})",
    )
    ax1.set_xticks(range(len(orden_temp)))
    ax1.set_xticklabels(etiquetas_temp, fontsize=10)
    ax1.set_title("Efecto Principal: Temperatura Inicial ($T_0$)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Nivel de Temperatura", fontsize=11)
    ax1.set_ylabel("Costo Promedio del Tour", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9.5)

    for i, val in enumerate(stats_temp["mean"]):
        ax1.annotate(
            f"{val:.2f}",
            (i, val),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9.5,
        )

    # Panel derecho: Efecto principal de Tamaño (n)
    stats_tam = df.groupby("tamano", observed=True)["costo"].agg(["mean", "count", "std"]).sort_index()
    ci95_tam = 1.96 * (stats_tam["std"] / np.sqrt(stats_tam["count"]))
    tamanos_idx = stats_tam.index.tolist()

    ax2.errorbar(
        range(len(tamanos_idx)),
        stats_tam["mean"],
        yerr=ci95_tam,
        fmt="-s",
        color="tab:orange",
        linewidth=2.2,
        markersize=8,
        capsize=5,
        label="Media ± IC 95%",
    )
    ax2.axhline(
        media_global,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label=f"Media global ({media_global:.2f})",
    )
    ax2.set_xticks(range(len(tamanos_idx)))
    ax2.set_xticklabels([f"n = {n}" for n in tamanos_idx], fontsize=10)
    ax2.set_title("Efecto Principal: Tamaño del Problema ($n$)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Tamaño del TSP (Número de Ciudades)", fontsize=11)
    ax2.set_ylabel("Costo Promedio del Tour", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=9.5)

    for i, val in enumerate(stats_tam["mean"]):
        ax2.annotate(
            f"{val:.2f}",
            (i, val),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9.5,
        )

    fig.suptitle(
        "Opción A — Gráfica de Efectos Principales (ANOVA de Dos Factores)",
        fontsize=13.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "opcion_a_efectos_principales.png", dpi=300)
    plt.show()

    # 2. GRÁFICA DE INTERACCIÓN (TEMPERATURA × TAMAÑO)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    estilos = [
        ("o", "-", "tab:green"),
        ("s", "--", "tab:blue"),
        ("^", "-.", "tab:red"),
    ]

    for idx, n in enumerate(tamanos_idx):
        sub = (
            df[df["tamano"] == n]
            .groupby("temperatura", observed=True)["costo"]
            .agg(["mean", "count", "std"])
            .reindex(orden_temp)
        )
        ci95 = 1.96 * (sub["std"] / np.sqrt(sub["count"]))
        marker, linestyle, color = estilos[idx % len(estilos)]

        ax.errorbar(
            range(len(orden_temp)),
            sub["mean"],
            yerr=ci95,
            marker=marker,
            linestyle=linestyle,
            color=color,
            linewidth=2.0,
            markersize=7.5,
            capsize=4,
            label=f"Tamaño n = {n}",
        )

    ax.set_xticks(range(len(orden_temp)))
    ax.set_xticklabels(etiquetas_temp, fontsize=10.5)
    ax.set_title(
        "Opción A — Gráfica de Interacción: Temperatura ($T_0$) × Tamaño ($n$)",
        fontsize=12.5,
        fontweight="bold",
    )
    ax.set_xlabel("Nivel de Temperatura Inicial ($T_0$)", fontsize=11)
    ax.set_ylabel("Costo Promedio del Tour (± IC 95%)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(title="Bloque (Instancia TSP)", fontsize=10, title_fontsize=10.5)

    fig.tight_layout()
    fig.savefig(out_dir / "opcion_a_interaccion.png", dpi=300)
    plt.show()

    print(f"\nArchivos generados en '{out_dir}':")
    print("  - resultados_opcion_a.csv")
    print("  - opcion_a_efectos_principales.png")
    print("  - opcion_a_interaccion.png")


if __name__ == "__main__":
    ejecutar_opcion_a()
