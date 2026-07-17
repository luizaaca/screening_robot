# Agente Screening Robot

Assistente de triagem clínica com estado, alimentado por LangGraph, busca de pacientes em SQLite e backends configuráveis para modelos clínicos.

## O que você pode fazer

- Perguntar como usar o assistente.
- Encontrar um paciente por número de segurança fictício ou por nome.
- Selecionar um paciente em uma lista enumerada de desambiguação.
- Limpar o contexto do paciente ativo.
- Descrever sintomas e solicitar condições prováveis ou exames relevantes.
- Enviar ou referenciar um vídeo para análise de expressão, postura e transcrição.

## Bons prompts para testar

- `Encontrar paciente Maria Silva`
- `Consultar paciente 12003456`
- `Paciente 55667788 tem fadiga e micção frequente`
- `Analise este vídeo com video_path=concepts_video/sample.mp4`
- `Limpar paciente ativo`

## Nota de demonstração

Se o banco de dados de pacientes estiver vazio, execute `python seed_demo_data.py` na raiz do projeto antes de testar os fluxos de busca.

## Isenção de responsabilidade

Este aplicativo serve apenas para prototipagem de software e validação de fluxo de trabalho. Ele não substitui avaliação médica profissional, diagnóstico ou atendimento de emergência.
