# Unsupervised Fraud Detection

Detecção de fraudes e perfis de consumo em base de cartões **sem rótulos** (`base04_cartao_alta_dimensao.csv`). Orientações do trabalho em [`docs/orientacoes.md`](docs/orientacoes.md).

## Estrutura

```
unsupervised-fraud-detection/
├── data/
│   ├── raw/            # base original (não editar)
│   └── processed/      # dados limpos/escalados gerados pelos notebooks
├── notebooks/          # 01_eda, 02_preprocessamento, 03_modelos, 04_resultados
├── src/                # funções reutilizáveis (pré-processamento, modelos, gráficos)
├── reports/figures/    # gráficos exportados para o artigo e slides
├── artigo/             # artigo (LaTeX/Overleaf ou PDF final)
├── apresentacao/       # slides da apresentação (15 min)
└── docs/               # orientações e anotações
```
