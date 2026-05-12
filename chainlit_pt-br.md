# Agente Screening Robot

Assistente de triagem clínica com estado, alimentado por LangGraph, SQLite para busca de pacientes e backends de modelos clínicos configuráveis.

## O que você pode fazer

- Perguntar como usar o assistente.
- Encontrar um paciente por número de segurança fictício ou por nome.
- Selecionar um paciente de uma lista de desambiguação enumerada.
- Limpar o contexto do paciente ativo.
- Descrever sintomas e solicitar condições prováveis ou exames relevantes.

## Bons prompts para testar

- `Encontrar paciente Maria Silva`
- `Consultar paciente 12003456`
- `Paciente 55667788 tem fadiga e micção frequente`
- `Limpar paciente ativo`

## Nota de demonstração

Se o banco de dados de pacientes estiver vazio, execute `python seed_demo_data.py` na raiz do projeto antes de testar os fluxos de busca.

## Isenção de responsabilidade

Este aplicativo é apenas para prototipagem de software e validação de fluxo de trabalho. Não substitui avaliação médica profissional, diagnóstico ou atendimento de emergência.