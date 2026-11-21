# Unsupervised Fraud Detection

Detecção não supervisionada de fraudes e segmentação de perfis de consumo em base de cartões de crédito sem rótulos (`base04_cartao_alta_dimensao.csv`), estruturada segundo a metodologia **KDD (*Knowledge Discovery in Databases*)**.

Repositório oficial: https://github.com/caioceo/unsupervised-fraud-detection

## Estrutura do Projeto

```
unsupervised-fraud-detection/
├── data/
│   ├── raw/            # base original (base04_cartao_alta_dimensao.csv - 5.000 clientes x 37 atributos)
│   └── processed/      # base processada com rótulos dos clusters, probabilidades GMM e lote de auditoria
├── notebooks/
│   └── notebooks.ipynb # pipeline KDD completo (Seleção, AED, Pré-processamento, PCA, K-Means, DBSCAN e GMM)
├── src/
│   └── util.py         # módulo de funções reutilizáveis para diagnóstico, gráficos, tratamento e modelagem
├── reports/
│   └── figures/        # gráficos de alta resolução exportados para o artigo e apresentação
├── artigo/
│   ├── sbc-template.tex # artigo científico em LaTeX (duas colunas)
│   └── referencias.bib  # referências bibliográficas em formato BibTeX
├── apresentacao/       # material para apresentação em aula (15 min)
└── docs/
    └── orientacoes.md  # orientações e barema da disciplina
```

## Resumo do Pipeline (Metodologia KDD)

1. **Seleção e Diagnóstico**: Base com 5.000 registros, 37 colunas (19 `float64`, 15 `int64`, 3 `object`) e 0 valores ausentes. Remoção do identificador `id_cliente`.
2. **Análise Exploratória de Dados (AED)**: Identificação de forte assimetria positiva nas variáveis financeiras (`limite_total_credito` até R$ 98,89 mil com mediana de R$ 7,89 mil) e caudas longas nos indicadores de risco (`qtd_saques_credito_12m`, `qtd_estornos_12m`, `pct_compras_madrugada` e `qtd_alteracoes_cadastrais_12m`).
3. **Pré-processamento e Transformação**:
   - **One-Hot Encoding** nas variáveis categóricas nominais (`tipo_lojista_predominante` e `canal_acesso_predominante`).
   - **Seleção de Atributos por Correlação**: Remoção de 8 variáveis numéricas redundantes com $|r| > 0,90$.
   - **Filtro de Outliers Extremos**: Critério de $4,0 \times \text{IQR}$, isolando 356 registros com distorção univariada severa para auditoria e retendo 4.644 clientes.
   - **Escalonamento com `RobustScaler`**: Padronização por mediana e IQR para evitar o esmagamento dos valores menores que ocorreria com `MinMaxScaler`.
   - **Redução de Dimensionalidade (PCA)**: Retenção de $64,43\%$ da variância total nos 3 primeiros componentes principais ortogonais.
4. **Modelagem Não Supervisionada**:
   - **K-Means (Baseline)**: $k = 4$ selecionado pelo Método do Cotovelo e máximo global de Calinski-Harabasz ($4.244,4$), Silhouette $= 0,4183$.
   - **DBSCAN (Detecção de Fraudes)**: $\varepsilon = 0,75$ e $\text{min\_samples} = 15$ (calibrados via curva *k-distance*), isolando **232 clientes anômalos (`cluster = -1`, $5,00\%$)** em regiões de baixa densidade.
   - **GMM (Segmentação de Clientes)**: $k = 4$ selecionado pelo mínimo de inflexão do BIC ($49.963,7$), modelando matrizes de covariância elípticas completas e identificando **193 clientes híbridos** em zona de transição probabilística ($P_{\max} < 0,80$).

