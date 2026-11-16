import math
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
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
# 2. TSP Y RECOCIDO SIMULADO CON 2-OPT (MISMA BASE QUE OPCIÓN A)
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
    """Ejecuta Recocido Simulado con enfriamiento exponencial y vecindad 2-opt."""
    rng = random.Random(semilla)
    n = len(matriz_dist)

    tour_actual = list(range(n))
    rng.shuffle(tour_actual)
    costo_actual = longitud_tour(tour_actual, matriz_dist)
    mejor_costo = costo_actual

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
# 4. ANOVA DE REGRESIÓN CON DESCOMPOSICIÓN DE FALTA DE AJUSTE Y ERROR PURO
# ==============================================================================
def calcular_anova_rsm(df_n: pd.DataFrame, y_hat: np.ndarray) -> dict[str, float]:
    """
    Descompone la variación total en Regresión, Residual, Falta de Ajuste y Error Puro.
    """
    y = df_n["costo"].to_numpy()
    y_bar = float(np.mean(y))
    N = len(y)
    m = df_n["temperatura"].nunique()  # 5 niveles
    p = 3  # 3 parámetros en modelo cuadrático (beta_0, beta_1, beta_2)

    ss_total = float(np.sum((y - y_bar) ** 2))
    ss_reg = float(np.sum((y_hat - y_bar) ** 2))
    ss_res = float(np.sum((y - y_hat) ** 2))

    # Error puro: variación dentro de cada nivel de temperatura
    ss_pe = 0.0
    for _, grupo in df_n.groupby("temperatura", observed=True):
        vals = grupo["costo"].to_numpy()
        ss_pe += float(np.sum((vals - np.mean(vals)) ** 2))

    ss_lof = max(0.0, ss_res - ss_pe)

    df_reg = p - 1           # 2
    df_res = N - p           # 27
    df_pe = N - m            # 25
    df_lof = m - p           # 2
    df_total = N - 1         # 29

    ms_reg = ss_reg / df_reg
    ms_res = ss_res / df_res
    ms_pe = ss_pe / df_pe
    ms_lof = ss_lof / df_lof

    f_reg = (ms_reg / ms_res) if ms_res > 1e-15 else np.nan
    p_reg = float(stats.f.sf(f_reg, df_reg, df_res)) if np.isfinite(f_reg) else np.nan

    f_lof = (ms_lof / ms_pe) if ms_pe > 1e-15 else np.nan
    p_lof = float(stats.f.sf(f_lof, df_lof, df_pe)) if np.isfinite(f_lof) else np.nan

    r2 = (ss_reg / ss_total) if ss_total > 1e-15 else np.nan

    return {
        "ss_reg": ss_reg,
        "df_reg": df_reg,
        "ms_reg": ms_reg,
        "f_reg": f_reg,
        "p_reg": p_reg,
        "ss_res": ss_res,
        "df_res": df_res,
        "ms_res": ms_res,
        "ss_lof": ss_lof,
        "df_lof": df_lof,
        "ms_lof": ms_lof,
        "f_lof": f_lof,
        "p_lof": p_lof,
        "ss_pe": ss_pe,
        "df_pe": df_pe,
        "ms_pe": ms_pe,
        "ss_total": ss_total,
        "df_total": df_total,
        "r2": r2,
    }


# ==============================================================================
# 5. EJECUCIÓN DE METODOLOGÍA DE SUPERFICIE DE RESPUESTA (OPCIÓN B)
# ==============================================================================
def ejecutar_opcion_b() -> None:
    cargar_env(".env")

    # Crear output/ y output/option_B/ en el mismo directorio raíz que este script
    out_dir = Path(__file__).resolve().parent / "output" / "option_B"
    out_dir.mkdir(parents=True, exist_ok=True)

    matricula = int(os.environ["MATRICULA"])
    r_fijo = float(os.environ["R_FIXED"])
    tamanos = [int(x.strip()) for x in os.environ["TSP_SIZES"].split(",")]

    # Configuración de los 5 niveles y réplicas: 5 en externos, 10 en el centro (30/tamaño)
    diseno_niveles = [
        ("Axial Bajo", float(os.environ["T_B_AXIAL_BAJO"]), 5),
        ("Bajo", float(os.environ["T_B_BAJO"]), 5),
        ("Centro", float(os.environ["T_B_CENTRO"]), 10),
        ("Alto", float(os.environ["T_B_ALTO"]), 5),
        ("Axial Alto", float(os.environ["T_B_AXIAL_ALTO"]), 5),
    ]

    print("=" * 84)
    print("OPCIÓN B: METODOLOGÍA DE SUPERFICIE DE RESPUESTA (RSM — 1 FACTOR CONTINUO)")
    print("=" * 84)
    print(f"Matrícula: {matricula} | r fijo = {r_fijo:.10f}")
    print(f"Bloques (n): {tamanos} | Corridas por tamaño: 30 (5×4 externos + 10 centro) | Total: 90")
    for nombre, temp_val, reps in diseno_niveles:
        print(f"  - {nombre:<12}: T_0 = {temp_val:8.4f} ({reps} réplicas)")
    print("-" * 84)

    # Mismas instancias por tamaño n que en la Opción A para comparabilidad directa
    instancias_dist = {}
    for n in tamanos:
        ciudades = generar_ciudades(n, semilla=matricula + n)
        instancias_dist[n] = calcular_matriz_distancias(ciudades)

    registros = []
    for n in tamanos:
        matriz_dist = instancias_dist[n]
        for idx_lvl, (nivel_nombre, T0, num_reps) in enumerate(diseno_niveles):
            for rep in range(1, num_reps + 1):
                semilla_corrida = matricula * 10_000 + n * 1_000 + idx_lvl * 100 + rep
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
                        "nivel": nivel_nombre,
                        "temperatura": T0,
                        "replica": rep,
                        "costo": costo,
                        "tiempo_s": tiempo_s,
                    }
                )

    df_total = pd.DataFrame(registros)
    orden_niveles = [d[0] for d in diseno_niveles]
    df_total["nivel"] = pd.Categorical(df_total["nivel"], categories=orden_niveles, ordered=True)
    df_total.to_csv(out_dir / "resultados_opcion_b.csv", index=False)

    # Analizar cada tamaño n de forma independiente
    for n in tamanos:
        df_n = df_total[df_total["tamano"] == n].copy()
        print("\n" + "=" * 84)
        print(f"ANÁLISIS RSM PARA TAMAÑO DE PROBLEMA n = {n} (N = {len(df_n)} observaciones)")
        print("=" * 84)

        resumen_n = (
            df_n.groupby(["nivel", "temperatura"], observed=True)["costo"]
            .agg(replicas="count", media="mean", std="std", mediana="median", min="min", max="max")
            .reset_index()
        )
        print("--- Resumen por Nivel de Temperatura ---")
        print(resumen_n.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        # Ajuste cuadrático: Costo = beta_0 + beta_1*T + beta_2*T^2
        T_vec = df_n["temperatura"].to_numpy()
        C_vec = df_n["costo"].to_numpy()
        varianza_nula = np.ptp(C_vec) < 1e-12

        beta2_np, beta1_np, beta0_np = np.polyfit(T_vec, C_vec, 2)
        modelo = ols("costo ~ temperatura + I(temperatura**2)", data=df_n).fit()
        y_hat = modelo.predict(df_n).to_numpy()

        beta0 = float(modelo.params["Intercept"])
        beta1 = float(modelo.params["temperatura"])
        beta2 = float(modelo.params["I(temperatura ** 2)"])
        p_beta1 = float(modelo.pvalues["temperatura"])
        p_beta2 = float(modelo.pvalues["I(temperatura ** 2)"])

        anova_rsm = calcular_anova_rsm(df_n, y_hat)

        print("\n--- Modelo Cuadrático Ajustado: Costo = β₀ + β₁·T + β₂·T² ---")
        print(f"  β₀ (Intercepto) = {beta0:12.6f}")
        print(f"  β₁ (Lineal T)   = {beta1:12.6f}  (p-value = {p_beta1:.6f})")
        print(f"  β₂ (Cuadrático) = {beta2:12.6f}  (p-value = {p_beta2:.6f})")
        print(f"  Verificación np.polyfit: β₀={beta0_np:.6f}, β₁={beta1_np:.6f}, β₂={beta2_np:.6f}")
        print(f"  R² = {anova_rsm['r2']:.6f}" if np.isfinite(anova_rsm["r2"]) else "  R² = N/A (varianza cero)")

        print("\n--- Tabla ANOVA de Regresión (con Falta de Ajuste y Error Puro) ---")
        print(f"{'Fuente':<22} {'SS':>12} {'df':>5} {'MS':>12} {'F':>10} {'p-value':>12}")
        print("-" * 77)
        print(
            f"{'Regresión (Modelo)':<22} {anova_rsm['ss_reg']:12.4f} {anova_rsm['df_reg']:5d} "
            f"{anova_rsm['ms_reg']:12.4f} {anova_rsm['f_reg']:10.4f} {anova_rsm['p_reg']:12.6f}"
        )
        print(
            f"{'Residual Total':<22} {anova_rsm['ss_res']:12.4f} {anova_rsm['df_res']:5d} "
            f"{anova_rsm['ms_res']:12.4f} {'':>10} {'':>12}"
        )
        print(
            f"{'  └ Falta de Ajuste':<22} {anova_rsm['ss_lof']:12.4f} {anova_rsm['df_lof']:5d} "
            f"{anova_rsm['ms_lof']:12.4f} {anova_rsm['f_lof']:10.4f} {anova_rsm['p_lof']:12.6f}"
        )
        print(
            f"{'  └ Error Puro':<22} {anova_rsm['ss_pe']:12.4f} {anova_rsm['df_pe']:5d} "
            f"{anova_rsm['ms_pe']:12.4f} {'':>10} {'':>12}"
        )
        print(
            f"{'Total':<22} {anova_rsm['ss_total']:12.4f} {anova_rsm['df_total']:5d}"
        )

        # Evaluación de óptimo T* y significancia
        T_optimo = None
        if varianza_nula or abs(beta2) < 1e-12:
            print("\n[Diagnóstico RSM]: Superficie completamente plana (costo constante en las 30 corridas).")
            print("  No existe curvatura (β₂ ≈ 0); cualquier T_0 alcanza el óptimo global.")
        else:
            T_optimo = -beta1 / (2.0 * beta2)
            T_min_exp, T_max_exp = float(T_vec.min()), float(T_vec.max())
            dentro_rango = T_min_exp <= T_optimo <= T_max_exp
            costo_pred_opt = beta0 + beta1 * T_optimo + beta2 * (T_optimo ** 2)

            print("\n[Diagnóstico RSM]:")
            if beta2 > 0:
                print(f"  - Concavidad: β₂ = {beta2:.6f} > 0 (parábola hacia arriba -> MÍNIMO de costo).")
                print(f"  - Temperatura óptima estimada: T* = -β₁ / (2β₂) = {T_optimo:.4f}")
                print(f"  - Costo predicho en T*: {costo_pred_opt:.4f}")
                print(
                    f"  - Ubicación de T*: {'DENTRO' if dentro_rango else 'FUERA (extrapolación)'} "
                    f"del rango experimental [{T_min_exp:.2f}, {T_max_exp:.2f}]."
                )
            else:
                print(
                    f"  - Concavidad: β₂ = {beta2:.6f} < 0 (parábola hacia abajo -> "
                    f"T* = {T_optimo:.4f} es MÁXIMO, no mínimo)."
                )

            print(
                f"  - Significancia curvatura (β₂): p = {p_beta2:.6f} "
                f"({'SIGNIFICATIVA (p < 0.05)' if p_beta2 < 0.05 else 'NO significativa (p >= 0.05)'})"
            )
            if np.isfinite(anova_rsm["p_lof"]):
                print(
                    f"  - Prueba de Falta de Ajuste: p = {anova_rsm['p_lof']:.6f} "
                    f"({'Modelo cuadrático adecuado (p >= 0.05)' if anova_rsm['p_lof'] >= 0.05 else 'Falta de ajuste significativa (p < 0.05)'})"
                )

        # Mediana vía Bootstrap (IC 95%) si el ANOVA de regresión es significativo
        anova_sig = np.isfinite(anova_rsm["p_reg"]) and (anova_rsm["p_reg"] < 0.05)
        if anova_sig:
            print(
                f"\n--- Mediana vía Bootstrap (10,000 remuestreos, IC 95%) para n = {n} "
                f"(ANOVA p = {anova_rsm['p_reg']:.6f} < 0.05) ---"
            )
            filas_boot = []
            for idx_b, ((lvl_name, t_val), grupo) in enumerate(
                df_n.groupby(["nivel", "temperatura"], observed=True)
            ):
                med, inf, sup = bootstrap_mediana_ic95(
                    grupo["costo"].to_numpy(), n_boot=10_000, semilla=matricula + n * 10 + idx_b
                )
                filas_boot.append(
                    {
                        "nivel": lvl_name,
                        "T0": t_val,
                        "replicas": len(grupo),
                        "mediana_boot": med,
                        "IC95_inf": inf,
                        "IC95_sup": sup,
                    }
                )
            print(pd.DataFrame(filas_boot).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        else:
            print(f"\n--- Bootstrap omitido para n = {n} (ANOVA de regresión no significativo) ---")

        # Generar gráfica individual para este tamaño n dentro de out_dir
        generar_grafica_rsm_tamano(
            df_n=df_n,
            modelo=modelo,
            n=n,
            beta0=beta0,
            beta1=beta1,
            beta2=beta2,
            T_optimo=T_optimo,
            varianza_nula=varianza_nula,
            out_dir=out_dir,
        )

    print(f"\nArchivos generados en '{out_dir}':")
    print("  - resultados_opcion_b.csv")
    for n in tamanos:
        print(f"  - opcion_b_rsm_n{n}.png")


# ==============================================================================
# 6. VISUALIZACIÓN INDIVIDUAL POR TAMAÑO (CURVA AJUSTADA + PUNTOS + IC 95%)
# ==============================================================================
def generar_grafica_rsm_tamano(
    df_n: pd.DataFrame,
    modelo,
    n: int,
    beta0: float,
    beta1: float,
    beta2: float,
    T_optimo: float | None,
    varianza_nula: bool,
    out_dir: Path,
) -> None:
    """Genera y guarda la curva RSM ajustada con puntos observados e IC 95% en out_dir."""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    T_min, T_max = df_n["temperatura"].min(), df_n["temperatura"].max()
    grid_T = np.linspace(T_min - 3.0, T_max + 3.0, 250)
    pred_df = pd.DataFrame({"temperatura": grid_T})
    pred_summary = modelo.get_prediction(pred_df).summary_frame(alpha=0.05)

    # 1. Banda de confianza del 95% de la curva de regresión
    ax.fill_between(
        grid_T,
        pred_summary["mean_ci_lower"],
        pred_summary["mean_ci_upper"],
        color="tab:blue",
        alpha=0.18,
        label="IC 95% de la curva ajustada",
    )

    # 2. Curva cuadrática ajustada
    ax.plot(
        grid_T,
        pred_summary["mean"],
        color="tab:blue",
        linewidth=2.3,
        label=rf"Ajuste cuadrático: $\hat{{C}}(T) = {beta0:.2f} {beta1:+.4f}T {beta2:+.6f}T^2$",
    )

    # 3. Puntos observados individuales (con ligera dispersión horizontal visual)
    rng_vis = np.random.default_rng(42 + n)
    jitter = rng_vis.uniform(-1.1, 1.1, size=len(df_n))
    ax.scatter(
        df_n["temperatura"] + jitter,
        df_n["costo"],
        color="dimgray",
        alpha=0.55,
        s=32,
        zorder=3,
        label="Observaciones individuales (30 corridas)",
    )

    # 4. Promedios por nivel de temperatura con IC 95%
    stats_lvl = (
        df_n.groupby(["nivel", "temperatura"], observed=True)["costo"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    ci95_lvl = 1.96 * (stats_lvl["std"] / np.sqrt(stats_lvl["count"]))
    ax.errorbar(
        stats_lvl["temperatura"],
        stats_lvl["mean"],
        yerr=ci95_lvl,
        fmt="D",
        color="tab:orange",
        ecolor="tab:orange",
        elinewidth=1.6,
        capsize=4,
        markersize=6.5,
        zorder=4,
        label="Media por nivel ± IC 95%",
    )

    # 5. Marcar T* si es un mínimo válido dentro o cerca del rango experimental
    if not varianza_nula and T_optimo is not None and beta2 > 0:
        if (T_min - 20.0) <= T_optimo <= (T_max + 20.0):
            c_opt = beta0 + beta1 * T_optimo + beta2 * (T_optimo ** 2)
            ax.axvline(
                T_optimo,
                color="tab:red",
                linestyle="--",
                linewidth=1.6,
                label=rf"Óptimo estimado $T^* = {T_optimo:.2f}$ ($\hat{{C}} = {c_opt:.2f}$)",
            )
            ax.scatter([T_optimo], [c_opt], color="tab:red", s=70, zorder=5)

    ax.set_title(
        rf"Opción B (RSM) — Superficie de Respuesta para Tamaño $n = {n}$",
        fontsize=12.5,
        fontweight="bold",
    )
    ax.set_xlabel(r"Temperatura Inicial ($T_0$)", fontsize=11)
    ax.set_ylabel("Costo del Tour", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9.2, loc="best")

    ruta_archivo = out_dir / f"opcion_b_rsm_n{n}.png"
    fig.tight_layout()
    fig.savefig(ruta_archivo, dpi=300)
    plt.show()
    print(f"Gráfica guardada: '{ruta_archivo}'")


if __name__ == "__main__":
    ejecutar_opcion_b()
