# Roteiro de Apresentação em Aula (15 Minutos Máximo — 15 Slides Numerados)

Este roteiro acompanha os slides de [apresentacao_slides.pdf](apresentacao_slides.pdf) (15 páginas numeradas de `01 / 15` a `15 / 15`). Cada slide foi dimensionado para **1 minuto em média**, totalizando exatamente **15 minutos** e cobrindo os 4 pilares avaliados no barema:
1. Visão geral do pipeline KDD, diagnóstico da Base 04 e pré-processamento;
2. Justificativa das escolhas técnicas (e por que descartamos alternativas);
3. Principais gráficos (PCA, Método do Cotovelo, Curva *k-distance*, BIC/AIC e Projeção 2D dos Clusters/Anomalias);
4. Conclusões e plano de ação prático antifraude para a operadora.

---

## Divisão de Falas por Integrante (Opção A: 3 Pessoas | Opção B: 5 Pessoas)

### Opção A — Equipe com 3 Integrantes (5 minutos e 5 slides por pessoa)
| Integrante | Slides | Tempo | Bloco Temático |
| :--- | :---: | :---: | :--- |
| **1. Thiago Domingos V. R. dos Santos** | **Slides 01 a 05** | `00:00 – 05:00` (5 min) | Introdução, Cenário da Operadora, Pipeline KDD, Diagnóstico da Base 04 e Análise Exploratória (AED). |
| **2. Enzo** | **Slides 06 a 10** | `05:00 – 10:00` (5 min) | Pré-processamento (*One-Hot Encoding*, Matriz de Correlação, Outliers $4,0 \times \text{IQR}$, `RobustScaler`, PCA) e Baseline K-Means. |
| **3. Caio** | **Slides 11 a 15** | `10:00 – 15:00` (5 min) | DBSCAN (Detecção de Fraudes), GMM (Segmentação Probabilística), Comparativo de Métricas/Perfis e Plano de Ação Antifraude. |

### Opção B — Equipe com 5 Integrantes (3 minutos e 3 slides por pessoa)
| Integrante | Slides | Tempo | Bloco Temático |
| :--- | :---: | :---: | :--- |
| **Integrante 1 (Thiago)** | **Slides 01 a 03** | `00:00 – 03:00` (3 min) | Contexto de negócio, desafio da Base 04 sem rótulos, arquitetura do Pipeline KDD e diagnóstico estrutural (5.000 clientes, 37 colunas, 0 nulos). |
| **Integrante 2 (Enzo)** | **Slides 04 a 06** | `03:00 – 06:00` (3 min) | A história dos dados na AED (cauda longa em limites, canais, lojistas e indicadores de risco) e *One-Hot Encoding* das variáveis nominais. |
| **Integrante 3 (Caio)** | **Slides 07 a 09** | `06:00 – 09:00` (3 min) | Poda de multicolinearidade ($|r| > 0,90$), remoção de outliers extremos ($4,0 \times \text{IQR}$), `RobustScaler` vs. `MinMaxScaler` e redução dimensional com PCA. |
| **Integrante 4** | **Slides 10 a 12** | `09:00 – 12:00` (3 min) | Estratégia dos 3 algoritmos, Baseline K-Means (Cotovelo $k=4$), DBSCAN para possíveis fraudes (232 anomalias) e GMM para clientes híbridos (193 em transição). |
| **Integrante 5** | **Slides 13 a 15** | `12:00 – 15:00` (3 min) | Como interpretar as métricas, projeção comparativa no espaço PCA, os 4 perfis comportamentais + anomalias e as 4 ações práticas antifraude. |

---

## Guia Slide a Slide (O que falar em cada página)

- **Slide 01 / 15 (00:00 – 00:45) — Capa e Objetivo**: Apresentar a equipe, o repositório no GitHub e a missão: identificar fraudes e segmentar perfis de consumo na base exclusiva `base04_cartao_alta_dimensao.csv` sem rótulos prévios.
- **Slide 02 / 15 (00:45 – 01:45) — O Desafio de Negócio e a Metodologia KDD**: Explicar por que adotamos o processo KDD em 5 etapas (Seleção, AED, Pré-processamento, Transformação e Mineração) e a separação entre funções reutilizáveis em `src/util.py` e experimentação em `notebooks/notebooks.ipynb`.
- **Slide 03 / 15 (01:45 – 02:45) — Diagnóstico Estrutural da Base 04**: Destacar 5.000 clientes, 37 colunas (19 `float64`, 15 `int64`, 3 `object`) e **0 valores nulos** (por isso não usamos imputação sintética por média ou KNN) e o descarte de `id_cliente`.
- **Slide 04 / 15 (02:45 – 03:45) — AED: Distribuições Financeiras e Hábito Categórico**: Mostrar a discrepância entre a mediana do limite (R$ 7,89 mil) e a cauda longa (até R$ 98,89 mil), além da concentração de 63,5% das compras em supermercado, combustível e restaurante, e 71,0% nos canais `contactless` e `web`.
- **Slide 05 / 15 (03:45 – 05:00) — AED: Maturidade da Conta e Indicadores de Risco/Fraude**: Explicar o comportamento de cauda longa nas variáveis de risco: `qtd_estornos_12m` (assimetria 3,03), `qtd_saques_credito_12m` (2,99), `pct_compras_madrugada` (2,05) e `qtd_alteracoes_cadastrais_12m` (até 11 ao ano).
- **Slide 06 / 15 (05:00 – 06:00) — Pré-processamento: Variáveis Categóricas e Multicolinearidade**: Justificar o **One-Hot Encoding** (variáveis nominais sem ordem; `LabelEncoder` criaria distância falsa) e o corte de **8 variáveis numéricas com $|r| > 0,90$** (como gastos de 3m e 12m com $r = 0,986$) para evitar peso duplicado na distância euclidiana.
- **Slide 07 / 15 (06:00 – 07:00) — Tratamento de Outliers Extremos ($4,0 \times \text{IQR}$)**: Explicar por que NÃO usamos $1,5 \times \text{IQR}$ (eliminaria 2.221 clientes ou 44,42% da carteira) nem Z-Score (efeito de mascaramento pela média inflada). O critério $4,0 \times \text{IQR}$ remove apenas 356 distorções severas (7,12%) e retém 4.644 clientes na base limpa.
- **Slide 08 / 15 (07:00 – 08:00) — Escalonamento com `RobustScaler`**: Explicar por que escolhemos mediana e IQR em vez de `MinMaxScaler` (que esmagaria os clientes de limite pequeno e médio perto de zero por causa do teto de R$ 98,9 mil) e em vez de `StandardScaler`.
- **Slide 09 / 15 (08:00 – 09:00) — Redução de Dimensionalidade com PCA**: Mostrar que os **3 primeiros componentes principais retêm 64,43% da variância total** (PC1: 35,1%, PC2: 17,3%, PC3: 12,1%), combatendo a maldição da dimensionalidade sem deformar densidades como fariam o t-SNE ou UMAP.
- **Slide 10 / 15 (09:00 – 10:00) — Arquitetura de Modelagem e Baseline K-Means ($k = 4$)**: Apresentar os 3 paradigmas e mostrar o gráfico do Método do Cotovelo (WCSS) e o pico global de Calinski-Harabasz ($4.244,4$) em $k = 4$ (Silhouette $= 0,4183$), explicando a limitação da fronteira rígida do K-Means.
- **Slide 11 / 15 (10:00 – 11:00) — Detecção de Possíveis Fraudes com DBSCAN**: Explicar a calibração da curva *k-distance* ($k = 15 \rightarrow \varepsilon = 0,75$), o isolamento de **232 clientes anômalos (`cluster = -1`, 5,00%)**, o Silhouette de $0,4462$ nos núcleos densos e a Razão de Isolamento de $2,21\times$.
- **Slide 12 / 15 (11:00 – 12:00) — Segmentação Probabilística com GMM ($k = 4$)**: Mostrar a curva BIC (mínimo de inflexão em $k = 4$, $\text{BIC} = 49.963,7$), a modelagem elíptica (`covariance_type="full"`) e os **193 clientes híbridos ($P_{\max} < 0,80$)** que pertencem parcialmente a dois perfis.
- **Slide 13 / 15 (12:00 – 13:00) — Projeção Comparativa no Espaço PCA e Como Ler as Métricas**: Comparar visualmente os 3 gráficos ($\text{PC1} \times \text{PC2}$) lado a lado e resumir a leitura rápida das métricas da tabela comparativa.
- **Slide 14 / 15 (13:00 – 14:00) — Os 4 Perfis de Consumo (GMM) e o Perfil de Fraude (DBSCAN `-1`)**: Detalhar quem é quem na carteira (G0: Alto Consumo Internacional; G1: Veteranos Digitais expostos a e-commerce/saques; G2: Varejo com 3,19 alterações cadastrais/ano; G3: Jovens com 13,82% de compras na madrugada; e Anomalias DBSCAN `-1` com 42,2 transações/mês).
- **Slide 15 / 15 (14:00 – 15:00) — Conclusão: Plano de Ação Prático Contra Fraudes**: Fechar traduzindo os achados técnicos nas 4 travas operacionais imediatas para a empresa (Step-Up Authentication no lote `-1`, quarentena de 24h pós-troca cadastral no G2/G3, cartão virtual dinâmico no G1 e gestão de limite guiada por probabilidades do GMM).
