# Imports Necessários
from skimage.measure import compare_ssim # skimage possui uma coleção de algorítmos para processamento de imagem
import argparse # Cria argumentos para passar na linha de comando
import imutils # Uma série de funções para facilitar trabalhar com imagens
import cv2


# Definindo argumentos para serem usados na linha de comando
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="first input image")
ap.add_argument("-s", "--second", required=True,
	help="second image")
args = vars(ap.parse_args())

imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])

# Convertendo a imagem para a escala de cinza (grayscale)
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# Calcular o Índice de Similaridade Estrutural (SSIM) entre as duas
# imagens, a imagem da diferença será retornada
(score, diff) = compare_ssim(grayA, grayB, full=True)
# convertemos a matriz em números inteiros não assinados de 8 bits no intervalo  [0, 255]
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# limiar a imagem da diferença, encontrar contornos para
# oobter a região da diferença das duas imagens
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# AGORA QUE JA TEMOS O LOCAL DA DIFERENÇA, VAMOS DESENHAR RETANGULOS NO LOCAL

# loop sobre o contorno
for c in cnts:
	# Computar a caixa em volta do contorno, então desenhar a caixa
	# lo local da diferença nas duas imagens
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
# Mostrar os outputs
cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
