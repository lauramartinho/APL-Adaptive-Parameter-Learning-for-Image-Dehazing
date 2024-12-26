import cv2
import numpy as np

# Carrega a imagem colorida
imagem = cv2.imread('/content/1898_0.85_0.08.jpg')

# Converte a imagem para escala de cinza
imagem_em_escala_de_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Aplica o thresholding com o método de thresholding adaptativo (cv2.ADAPTIVE_THRESH_MEAN_C)
# com um tamanho de bloco de 11x11 e uma constante de 2 subtraída do valor médio
mascara = cv2.adaptiveThreshold(imagem_em_escala_de_cinza, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Inverte a máscara (para que áreas brancas se tornem 0 e áreas pretas se tornem 255)
mascara_invertida = cv2.bitwise_not(mascara)

# Cria uma cópia da imagem original
imagem_colorida_mascarada = imagem.copy()

# Pinta as áreas onde a máscara é preta com a cor preta na imagem mascarada
imagem_colorida_mascarada[mascara_invertida == 255] = [0, 0, 0]

# Mostra a imagem com a máscara aplicada
cv2_imshow(imagem_colorida_mascarada)
cv2.waitKey(0)
cv2.destroyAllWindows()
