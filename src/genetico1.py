"""
algoritmo genetico aplicado no problema de escolher específicas quantidades de N produtos de tipos 
diferentes em L lojas de maneira a se obter menor custo, onde custo é calculado como a soma do custo 
de preço com o custo de frete (dependendo do custo de preço, o frete passa a ser gratuito)


Fij vale o frete unitario do produto j na loja i, e se o valor gasto (custo de preço e nao de frete) 
numa loja i atingir Vi, o frete total se torna gratuito
"""
import math
from sys import argv
import pandas as pd
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import rcParams
from random import shuffle
import pylab as plt
#set the plot figure size
rcParams["figure.figsize"] = 10, 5
#%matplotlib inline

class Genetico(object):

	"""
	Parametros do algoritmo genetico
	--------------------------------------
	crc: float
	(chance de reproducao do casal)
	chance de reproducao com sucesso de um determinado casal emparelhado
	
	alpha: float
	parametro usado no calculo da funcao de cruzamento aritmetico
	
	n_iter: int
	numero maximo de iteracoes
	
	cmm: float
	(chance de mutacao do membro)
	chance de mutacao de um membro da populacao, seus dois atributos sofrem mutacao

	mut_interval: float
	distancia maxima de mutacao do valor original, desta maneira o novo valor sera entre 
	[valor + mut_interval, valor - mut_interval]	

	k: int
	indica quantos dos melhores membros serão clonados para a próxima geração
	--------------------------------------


	Constantes do problema de minimizacao do custo de compra
	--------------------------------------
	N: int
	numero de tipos de produtos diferentes (cada um deles deve ser comprado em determinada quantidade)

	L: int
	numero de lojas que dispomos para comprar os produtos

	V: array-like	shape(L)
	Vi - valor de custo de preço a ser atingido na compra da loja i para se ter custo de frete gratuito

	Q: array-like 	shape(N)
	Qj - quantidade do produto j que deve ser comprada

	P: array-like 	shape(L,N)
	Pij - custo de preço unitario do produto j na loja i

	F: array-like 	shape(L,N)
	Fij - frete unitario do produto j na loja i
	--------------------------------------

	Variaveis
	--------------------------------------
	X: array-like 	shape(L,N)
	Xij - quantidade do produto j a ser comprado na loja i
	--------------------------------------
	"""

	def __init__(self, pop_size=100, crc=0.7, alpha=0.5, n_iter=1000, cmm=0.01, mut_interval=10.0, k=1):
		self.pop_size = pop_size
		self.crc = crc
		self.alpha = alpha
		self.n_iter = n_iter
		self.cmm = cmm
		self.mut_interval = mut_interval
		self.k = k
		self.medias = []
		self.melhores = []
		self.piores = []		
		self.N = 10
		self.L = 5
		'''self.V = np.random.randint(1,10,size=(self.L))'''
		self.V = np.array([5, 3, 5, 8, 4])
		print 'V', self.V
		'''self.P = np.random.randint(1,50,size=(self.L,self.N))'''
		self.P = np.array([[31, 27, 19, 25, 43, 33,  8, 30, 37,  8], [35, 20, 10, 26, 19, 23,  3, 27, 31, 37], [ 6, 13, 14, 29, 29, 49, 49,  4, 40,  8], [24,  7,  3, 30, 18, 38, 47, 37, 43, 14], [14, 39, 41, 46, 13, 36, 13, 39, 49, 13]])
		print 'P', self.P
		'''self.F = np.random.randint(1,10,size=(self.L,self.N))'''
		self.F = np.array([[9, 9, 1, 6, 3, 9, 1, 9, 3, 4], [9, 2, 1, 1, 6, 6, 8, 8, 8, 9], [9, 7, 9, 3, 5, 8, 6, 7, 2, 2], [1, 3, 3, 8, 6, 2, 1, 9, 7, 5], [7, 3, 8, 6, 5, 1, 8, 6, 3, 9]])
		print 'F', self.F
		'''self.Q = np.random.randint(1,10,size=(self.N))'''
		self.Q = np.array([1, 7, 7, 5, 7, 1, 8, 4, 1, 6])
		print 'Q', self.Q
		

	def calcula_custo_total(self, X):

		custo_total = 0.0
		"""para cada loja"""
		for i in range(self.L):
			custo_preco = 0.0
			custo_frete = 0.0
			"""para cada produto"""
			for j in range(self.N):
				custo_preco += X[i][j]*self.P[i][j]
				custo_frete += X[i][j]*self.F[i][j]

			if custo_preco >= self.V[i]:				
				custo_total += custo_preco
			else:	
				custo_total += custo_preco + custo_frete

		return custo_total

	"""
	funcao de avaliacao escolhida - funcao de custo
	Custo total = custo preço + custo frete
	custo preço = sum (1 <= i <= L) [sum (1 <= j <= N) Xij * Pij]
	custo frete = sum (1 <= i <= L) [sum (1 <= j <= N) Xij * Fij]
	sobre a funcao de avaliacao, ela inicialmente era de minimizacao, entao tivemos de modifica-la para
	que a area na roleta de um individuo seja proporcional ao seu fitness (e queremos que os de menor valor
	na funcao original de minimizacao tenham maior area na roleta e maior probabilidade de cruzamento) e 
	por isso deslocamos a funcao e invertemos seu valor. Nosso problema agora é de maximizacao; o melhor 
	individuo possuira o maior fitness; o pior tera o menor fitness
	"""
	def avaliacao(self, X):
		custo_total = self.calcula_custo_total(X)
		return 7000 - custo_total

	"""
	essa funcao é capaz de escolher um membro dentro da populacao, em que o membro com maior avaliacao possui maior chance de ser escolhido
	"""
	def girar_roleta(self, soma, avaliacoes):
		roleta = np.random.random() * soma 
		for i in range(self.pop_size):	
			roleta -= avaliacoes[i]
			if roleta <= 0.0:
				return i

	def swap(self,vetor,i,j):
		temp = vetor[i]
		vetor[i] = vetor[j]
		vetor[j] = temp
		return vetor

	"""
	metodo que ordena a populacao e suas avaliacoes, em ordem decrescente de avaliacao
	"""
	def insertion_sort(self, avaliacoes, populacao):
		for i in range(self.pop_size-1):
			if(avaliacoes[i+1]>avaliacoes[i]):
				j = i
				while j>=0 and avaliacoes[j+1]>avaliacoes[j]:
					self.swap(avaliacoes,j,j+1)
					self.swap(populacao,j,j+1)
					j -= 1
		return avaliacoes

	def executar(self):
		"""
		populacao de tamanho self.pop_size, cada membro possui matriz L x N de valores aleatorios que variam entre 0 e 10, inclusive esses
		"""
		populacao = np.random.randint(0,10+1,size=(self.pop_size,self.L,self.N))
		k_melhores = np.zeros((self.k,self.L,self.N))
		filhos = np.zeros((self.pop_size,self.L,self.N))
		avaliacoes = np.zeros(self.pop_size)

		"""
		para n iteracoes, faça:		
		"""
		for k in range(self.n_iter):	
			"""
 			etapa 1 - avaliacao
			
			para cada membro, calcular o fitness de cada individuo de acordo com a funcao de avaliacao
			"""
			soma = 0.0
			media = 0.0
			pior = self.avaliacao(populacao[0])
			melhor = pior
			for n in range(self.pop_size):
				for j in range(self.N):
					qtde_atual_prod_j = 0
					for i in range(self.L):
						qtde_atual_prod_j += populacao[n][i][j]
					"""se a quantidade do produto j é menor que a desejada, corrigir"""		
					while qtde_atual_prod_j < self.Q[j]:
					 	"""pega loja aleatoria"""				
						loja = np.random.randint(0,self.L)
						"""pega quantidade aleatoria do produto j na loja"""
						o_quanto_falta_pegar = self.Q[j] - qtde_atual_prod_j
						adicao_prod_j_na_loja = np.random.randint(0,o_quanto_falta_pegar+1)
						populacao[n][loja][j] += adicao_prod_j_na_loja
						"""atualiza quantidade atual do produto j"""
						qtde_atual_prod_j += adicao_prod_j_na_loja

					"""se a quantidade do produto j é maior que a desejada, corrigir"""		
					while qtde_atual_prod_j > self.Q[j]:
					 	"""pega loja aleatoria"""				
						loja = np.random.randint(0,self.L)
						"""tira quantidade aleatoria do produto j na loja, desde que o valor nao fique negativo"""
						o_quanto_falta_tirar = qtde_atual_prod_j - self.Q[j]
						remocao_prod_j_na_loja = min(np.random.randint(0,o_quanto_falta_tirar+1), populacao[n][loja][j])
						populacao[n][loja][j] -= remocao_prod_j_na_loja
						"""atualiza quantidade atual do produto j"""
						qtde_atual_prod_j -= remocao_prod_j_na_loja				

				avaliacoes[n] = self.avaliacao(populacao[n])
				"""calcular a melhor, a pior e a media de avaliacoes"""
				soma += avaliacoes[n]
				if avaliacoes[n] > melhor:
					melhor = avaliacoes[n]
				if avaliacoes[n] < pior:
					pior = avaliacoes[n]
		
			media = soma/self.pop_size	
			self.medias.append(media)
			self.melhores.append(melhor)
			self.piores.append(pior)

			'''print("populacao",populacao)'''
			'''print("avaliacoes",avaliacoes)'''
			'''input("aperte ENTER para prosseguir")'''
			'''input("aperte ENTER para prosseguir")'''			
			avaliacoes = self.insertion_sort(avaliacoes,populacao)
			'''print("populacao ordem decrescente de avaliacao",populacao)'''
			'''print("avaliacoes ordenadas",avaliacoes)'''
			'''input("aperte ENTER para prosseguir")'''
			for n in range(self.k):
				k_melhores[n] = populacao[n]
			'''print("k melhores",k_melhores)'''
			'''input("aperte ENTER para prosseguir")'''

			'''print (k+1),'\b-esima iteracao - melhor', populacao[0]
			print 'avaliacao do melhor', avaliacoes[0]'''

			"""
			sao muitas dimensoes, impossivel plotar tudo no mesmo grafico, o que podemos fazer é 
			imprimir membros selecionados da populacao e exibir suas avaliacoes
			"""
			if  k == self.n_iter-1:				
				plt.subplot(121)
				plt.ylabel('comparacao das avaliacoes: maxima, menor e media')
				plt.xlabel('iteracoes')
				axis = range(k+1)
				plt.plot(axis, self.medias, 'r--', axis, self.melhores, 'bs', axis, self.piores, 'g^')
				plt.show()

				print (k+1),'\b-esima Iteracao'
				print 'Melhor avaliacao: ', melhor
				print 'Pior avaliacao: ', pior
				print 'Media: ', media, '\n'		
				print 'gene do melhor = matriz de elementos Xij (loja i, produto j):'
				print populacao[0]

			"""
 			etapa 2 - selecao

	 		a populacao sera mantida a cada iteracao como 100
	 		rodamos a roleta de selecao 100 vezes, e a cada duas rolagens emparelha-se um casal; 
			individuos com maior area na roleta tem maior chance de serem escolhidos
			calculamos uma chance do casal reproduzir, que pode ser 70% de chance de reproducao com sucesso
			"""			
			
			filhos.fill(-1)

			for i in range(int(self.pop_size/2)):
				pai_1 = self.girar_roleta(soma, avaliacoes)
				pai_2 = self.girar_roleta(soma, avaliacoes)
				chance_reproducao = np.random.random()
				if chance_reproducao >= self.crc:
					"""
					etapa 3 - reproducao

					para cada casal escolhido e com chance de reproducao maior que o valor escolhido anteriormente, nós definimos 2 filhos
					cada um dos 2 filhos é definido pela formula, que define o i-esimo atributo do filho de acordo com o i-esimo atributo 
					do pai 1 e com o i-esimo atributo do pai 2
					F1[i] = alpha * P1[i] + (1-alpha) * P2[i]
					F2[i] = alpha * P2[i] + (1-alpha) * P1[i]
					onde alpha é valor aleatorio entre 0 e 1
					filho 1 substitui pai 1
					filho 2 substitui pai 2
					individuos que nao reproduziram sao mantidos na populacao, como se tivessem sido substituidos por filhos clones
					"""
					'''print("pai 1",populacao[pai_1])'''
					'''print("avaliacao pai 1",avaliacoes[pai_1])'''
					'''print("pai 2",populacao[pai_2])'''
					'''print("avaliacao pai 2",avaliacoes[pai_2])'''
					'''input("aperte ENTER para prosseguir")'''
					
					for i in range(self.L):
						for j in range(self.N):
							filhos[pai_1][i][j] = math.ceil(self.alpha*populacao[pai_1][i][j] + (1-self.alpha)*populacao[pai_2][i][j])
							filhos[pai_2][i][j] = math.ceil(self.alpha*populacao[pai_2][i][j] + (1-self.alpha)*populacao[pai_1][i][j])

					'''print("filho 1",filhos[pai_1])'''
					'''print("filho 2",filhos[pai_2])'''
					'''input("aperte ENTER para prosseguir")'''

			for n in range(self.pop_size):
				if filhos[n][0][0] >= 0:
					populacao[n] = filhos[n]
					avaliacoes[n] = self.avaliacao(populacao[n])

			"""
			4 - mutacao - mutacao CREEP
			cada membro tem 1% de chance de sofrer mutacao,
			caso aconteça, seus dois atributos sofrem uma modificacao em que os novos valores passam a ser
			novo x1 = entre [x1 + mut_interval, x1 - mut_interval]	
			novo x2 = entre [x2 + mut_interval, x2 - mut_interval]	
			"""
			for n in range(self.pop_size):
				chance_mutacao = np.random.random()
				if chance_mutacao > self.cmm:
					for i in range(self.L):
						for j in range(self.N):
							valor = populacao[n][i][j]
							mutacao = np.random.randint(-self.mut_interval, self.mut_interval+1)
							if n < 20:
								mutacao = math.ceil(mutacao * (1-avaliacoes[i]/melhor))
							if valor + mutacao >= 0:
								populacao[n][i][j] = valor + mutacao
				if n >= self.pop_size - 20:
					for i in range(self.L):
						for j in range(self.N):
							populacao[n][i][j] = np.random.randint(0,11)		

			"""
			agora nos substituimos os k piores individuos da proxima geracao pelos k melhores da anterior	
			"""
			avaliacoes = self.insertion_sort(avaliacoes,populacao)
			for n in range(self.k):
				populacao[self.pop_size-self.k+n] = k_melhores[n]

			'''print("proxima geracao apos mutacao",populacao)'''
			'''print("avaliacoes da proxima geracao",avaliacoes)'''
			'''input("aperte ENTER para prosseguir")'''	

genetico = Genetico()
genetico.executar()