# 🧬 Breast Cancer Detection with K-Nearest Neighbors

Projeto de Machine Learning para classificação de tumores como malignos ou benignos utilizando o algoritmo KNN (K-Nearest Neighbors).

---

## 📋 Descrição do Projeto

Este projeto realiza uma análise completa, desde a exploração dos dados até a otimização de hiperparâmetros.

O dataset utilizado é o **Breast Cancer Wisconsin Diagnostic**, disponível na biblioteca Scikit-learn, contendo características extraídas de imagens de biópsias.

---

## ⚙️ Etapas do Projeto

- Carregamento e exploração dos dados com `load_breast_cancer`
- Divisão dos dados (80% treino / 20% validação)
- Treinamento inicial com `k = 3`
- Teste de múltiplos valores de `k` (1 a 100)
- Avaliação da acurácia para escolha do melhor modelo
- Testes com diferentes `random_state` para verificar estabilidade

---

## 📊 Resultados

- Tamanho do treino: 455 amostras  
- Tamanho da validação: 114 amostras  
- Acurácia com k=3: ~0.9474  
- Média de acurácia (random states): ~0.9323  

O projeto também gera um gráfico mostrando a variação da acurácia conforme o valor de `k`.

---

