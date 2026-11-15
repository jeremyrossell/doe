import random

# 1. Fijar semilla única con la matrícula
matricula = "2173396"
random.seed(matricula)
r = random.random()

# 2. Definir rango de temperatura inicial [T_min, T_max]
T_min = 1.0
T_max = 150.0
rango = T_max - T_min

# 3. Calcular fracciones y temperaturas para Opción A (3 niveles)
fracciones_A = {
    "Bajo": 0.3 * r,
    "Medio": 0.3 * r + 0.3,
    "Alto": 0.3 * r + 0.6,
}
niveles_A = {k: T_min + f * rango for k, f in fracciones_A.items()}

# 4. Calcular fracciones y temperaturas para Opción B (5 niveles)
fracciones_B = {
    "Axial Bajo": 0.2 * r,
    "Bajo": 0.2 * r + 0.2,
    "Centro": 0.2 * r + 0.4,
    "Alto": 0.2 * r + 0.6,
    "Axial Alto": 0.2 * r + 0.8,
}
niveles_B = {k: T_min + f * rango for k, f in fracciones_B.items()}

print(f"Matrícula: {matricula}")
print(f"Valor fijo r = {r:.10f}\n")

print("--- OPCIÓN A (3 Niveles | 3 x 3 x 10 = 90 corridas) ---")
for nombre, frac in fracciones_A.items():
    print(f"  {nombre:<12} | Fracción: {frac:.6f} | T_0 = {niveles_A[nombre]:.4f}")

print("\n--- OPCIÓN B (5 Niveles | 5 x 3 x 10 = 150 corridas) ---")
for nombre, frac in fracciones_B.items():
    print(f"  {nombre:<12} | Fracción: {frac:.6f} | T_0 = {niveles_B[nombre]:.4f}")