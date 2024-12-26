def clip_predictions(predictions, lower_limit, upper_limit):
    return np.clip(predictions, lower_limit, upper_limit)

# Faz previsões em novas imagens

DIR_IMG = '/content/drive/MyDrive/DEFFOGGING/VIDEO_1_RESULTS/nanchang_18_8_8_1_17421_1977.png'
imagem_nova = Image.open(DIR_IMG)
imagem_nova = imagem_nova.resize((64, 64))  # Redimensiona a imagem para o mesmo tamanho das imagens de treinamento
imagem_nova_array = np.array(imagem_nova) / 255.0  # Normaliza os valores dos pixels
imagem_nova_array = np.expand_dims(imagem_nova_array, axis=0)  # Adiciona uma dimensão extra para a amostra única

predicao = modelo.predict(imagem_nova_array)

predicao_clip = clip_predictions(predicao, 0.8, 2.0)

cci_pred = round(predicao_clip[0][0], 1 )
gamma_pred = round(predicao_clip[0][1],1 )
exp_pred = round(predicao_clip[0][2],1 )

img = cv2.imread(DIR_IMG)
restored = dehaze(img, cci_pred, gamma_pred, exp_pred)
# restored2 = dehaze(img, 1.0, 1.0, 1.0)

print("CCI:", cci_pred, "Gamma:", gamma_pred, "Exposicao:", exp_pred)

cv2_imshow(img)
cv2_imshow(restored)
# cv2_imshow(restored2)


import cv2
from google.colab.patches import cv2_imshow
indice_teste = 14 # Índice do exemplo de teste a ser selecionado
imagem_teste = X_teste[indice_teste]
rotulo_real = y_teste[indice_teste]

# Faz a previsão do exemplo de teste
imagem_teste_array = np.expand_dims(imagem_teste, axis=0)
predicao_teste_1 = modelo.predict(imagem_teste_array)[0]

predicao_teste = clip_predictions(predicao_teste_1, 0.8, 2.0)


imagem_teste_bgr = cv2.cvtColor((imagem_teste * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
# Exibe a imagem de teste

cci_real = round(rotulo_real[0] , 1)
gamma_real = round(rotulo_real[1] , 1)
exp_real = round(rotulo_real[2] , 1)
print(rotulo_real)

cci_predito = round(predicao_teste[0] , 1)
gamma_predito = round(predicao_teste[1] , 1)
exp_predito = round(predicao_teste[2] , 1)
print(predicao_teste)

print("Valores Reais:")
print("CCI: {:.1f}".format(cci_real), "Gamma: {:.1f}".format(gamma_real), "Exposicao: {:.1f}".format(exp_real))

cv2_imshow(imagem_teste_bgr)

print("Valores Preditos:")
print("CCI: {:.1f}".format(cci_predito), "Gamma: {:.1f}".format(gamma_predito), "Exposicao: {:.1f}".format(exp_predito))

restored = dehaze(imagem_teste_bgr, cci_predito, gamma_predito, exp_predito)
cv2_imshow(restored)
