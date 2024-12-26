X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(
    imagens, rotulos, test_size=0.2, random_state=42
)
#normalizar pixels:
X_treinamento = X_treinamento / 255.0
X_teste = X_teste / 255.0

num_folds = 5

acuracias = []

kf = KFold(n_splits=num_folds, shuffle=True)

def criar_modelo():
    modelo = Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3)
    ])

    modelo.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return modelo


modelo = criar_modelo()
print(modelo.summary())

modelo = KerasRegressor(build_fn=criar_modelo)

parametros = {
    'batch_size': [16,64],
    'epochs': [10,15]
}

grid_search = GridSearchCV(estimator=modelo, param_grid=parametros, scoring='neg_mean_absolute_error', cv=kf)
grid_search.fit(imagens, rotulos)

resultados = grid_search.cv_results_
melhores_parametros = grid_search.best_params_
melhor_acuracia = -grid_search.best_score_

print("\nMelhores parâmetros:")
print(melhores_parametros)

print("Acurácia média:")
print(melhor_acuracia)
print("")

print("Todos os Resultados:\n")
for indice, parametro_combinacao in enumerate(resultados['params']):
    acuracia = -resultados['mean_test_score'][indice]
    print(f"Combinação de parâmetros: {parametro_combinacao}")
    print(f"Acurácia: {acuracia}")
    print("")


modelo = Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3)
])


# Compila o modelo
modelo.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Treina o modelo
historicofinal = modelo.fit(X_treinamento, y_treinamento, epochs=15, batch_size=16, verbose=1)

# Avalia o modelo no conjunto de teste
perda, acuracia = modelo.evaluate(X_teste, y_teste, verbose=0)
print(f"Acurácia no conjunto de teste: {acuracia} ({acuracia * 100}%)")

# Plotar gráfico de acurácia
plt.plot(historicofinal.history['accuracy'])
plt.title('Acurácia do Modelo')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.show()

# Plotar gráfico de perda
plt.plot(historicofinal.history['loss'])
plt.title('Perda do Modelo')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.show()
