DIR = '/content/drive/MyDrive/UWE/Laura_ImpactLab/Datasets/Dataset_Tahi/' #diretório das imagens
DIR_results = '/content/drive/MyDrive/UWE/Laura_ImpactLab/Datasets/Tahi_Results/'
arquivo = '/content/drive/MyDrive/tahi.txt'
melhores_resultados = [] #vetor que receberá os resultados

#percorrer pasta de arquivos
for filename in sorted(os.listdir(DIR)):
  resultados=[] #lista que recebe a string contendo os dados (imagem, parâmetros e resultados qualidade)
  caminho_img = f'{DIR}/{filename}'

#   img = cv2.imread(caminho_img)
  imgem = cv2.imread(caminho_img)
  img = resize(imgem, 50)

  print(filename)

  #parâmetros que serão testados
  cci = [1.0, 2.0] #intensidade de correcao de cor
  gamma = [0.8, 0.9] #intensidade do gamma
  exposicao =[0.8, 0.9, 1.0] #intensidade do alfa da combinação

  for i in range(len(cci)):
    for j in range(len(gamma)):
      for m in range(len(exposicao)):
        # for n in range(len(beta)):

        #image, cci, gamma, alfa, beta, sigma=0.5, intensidade=1.0, brightness=5, contrast=10
        restored = dehaze(img, cci[i], gamma[j], exposicao[m])

        #metricas de qualidade
        niq = getNIQE(restored)
        briq = getBRISQUE(restored)

        #psnr = peak_signal_noise_ratio(img, restored)
        #ssim = structural_similarity(img, restored,  multichannel=True)

        string = str(filename), str(cci[i]), str(gamma[j]), str(exposicao[m]), "{:.4f}".format(niq), "{:.4f}".format(briq)
        resultados.append(string)

  maior, indice = maior_valor(resultados)

  if(maior != -1 and indice != -1):
    melhores_resultados.append(resultados[indice])

  print(f'maior {maior}, indice {indice}, vetor[indice] {resultados[indice]}')

  my_str = '_'.join(resultados[indice]) .replace('0', '').replace('.', '').replace('png', "") + '.png'
  print(my_str)
  cv2.imwrite(f'{DIR_results}{str(my_str)}', restored)
escreve(arquivo, melhores_resultados)
