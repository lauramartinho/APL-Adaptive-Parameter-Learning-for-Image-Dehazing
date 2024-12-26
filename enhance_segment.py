import cv2
import numpy as np


# Carrega a imagem colorida
imagem = cv2.imread('/content/drive/MyDrive/boa.jpg')

# Converte a imagem para escala de cinza
imagem_em_escala_de_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Aplica o thresholding com um valor de threshold maior (170 neste caso)
_, mascara = cv2.threshold(imagem_em_escala_de_cinza, 170, 255, cv2.THRESH_BINARY)

# Inverte a máscara (para que áreas brancas se tornem 0 e áreas pretas se tornem 255)
mascara_invertida = cv2.bitwise_not(mascara)

# Aplica a sua função de correção de cor apenas nas áreas definidas pela máscara
areas_corrigidas = mascaraNitidez(imagem, 1.0, 3.0, kernel=(5, 5), threshold=0) # Ajuste o valor de intensidade conforme necessário

# Mantém as áreas onde a máscara é preta inalteradas e substitui as áreas brancas pela imagem corrigida
imagem_colorida_mascarada = cv2.bitwise_and(imagem, imagem, mask=mascara_invertida)
imagem_colorida_mascarada += cv2.bitwise_and(areas_corrigidas, areas_corrigidas, mask=mascara)

# Mostra a imagem com a máscara e o filtro aplicado
cv2_imshow( imagem_colorida_mascarada)
cv2.waitKey(0)
cv2.destroyAllWindows()
