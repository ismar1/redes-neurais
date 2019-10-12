# -*- coding: utf-8 -*-
"""
@author: Ismar
"""

import numpy as np

def sigmoid(soma):
  return 1 / (1 + np.exp(-soma))

def sigmoidDerivada(sig):
  return sig * (1 - sig)

# Operador XOR
entradas = np.array([[0,0],
                     [0,1],
                     [1,0],
                     [1,1]])

saidas = np.array([[1],[0],[0],[1]])

pesos0 = 2 * np.random.random((2,3)) - 1
pesos1 = 2 * np.random.random((3,1)) - 1

epocas = 10000
taxaAprendizagem = 0.5
momento = 1

for j in range(epocas):
  # Somatório das entradas com os pesos
  camadaEntrada = entradas
  somaSinapse0 = np.dot(camadaEntrada, pesos0)
  # Sigmoid da camada oculta
  camadaOculta = sigmoid(somaSinapse0)
  
  # Somatório da camada oculta com os pesos
  somaSinapse1 = np.dot(camadaOculta, pesos1)
  # Sigmoid da camada saída
  camadaSaida = sigmoid(somaSinapse1)
  
  # Erro
  erroCamadaSaida = saidas - camadaSaida
  mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
  print("Erro: " + str(mediaAbsoluta))
  
  # Derivada e delta saída
  derivadaSaida = sigmoidDerivada(camadaSaida)
  deltaSaida = erroCamadaSaida * derivadaSaida
  
  # Delta da camada oculta
  pesos1Transposta = pesos1.T
  deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
  deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)
  
  # Atualização dos pesos da saída para camada oculta
  camadaOcultaTransposta = camadaOculta.T
  pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
  pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)
  
  # Atualização dos pesos da camada oculta para as entradas
  camadaEntradaTransposta = camadaEntrada.T
  pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
  pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)
  
# Arredondamento dos resultados
for s in range(len(camadaSaida)):
  if camadaSaida[s] < 0.5:
    camadaSaida[s] = 0
  else:
    camadaSaida[s] = 1.0

print('\nChance de acerto: {0:.1f}%'.format((1 - mediaAbsoluta) * 100))
  
  
  
  
  
  
  
  
  
  
  
  
  