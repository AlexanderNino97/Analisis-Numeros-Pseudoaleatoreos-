def congruencia_lineal(semilla, a, c, m, n):
    numeros = []
    X = semilla
    for _ in range(n):
        X = (a * X + c) % m
        numeros.append(X)
    return numeros

# Solicitar parámetros al usuario
semilla = int(input("Ingrese la semilla inicial (X0): "))
a = int(input("Ingrese el multiplicador (a): "))
c = int(input("Ingrese el incremento (c): "))
m = int(input("Ingrese el módulo (m): "))
n = int(input("Ingrese la cantidad de números pseudoaleatorios a generar (n): "))

# Generar números pseudoaleatorios
numeros_aleatorios = congruencia_lineal(semilla, a, c, m, n)

# Imprimir los números generados
print("\nNúmeros pseudoaleatorios generados:")
for i, num in enumerate(numeros_aleatorios):
    print(f"X_{i+1} = {num}")