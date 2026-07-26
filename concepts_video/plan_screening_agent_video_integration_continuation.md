# Plano de continuidade: refinamento do `screening_agent` com video

Atualizado em: 2026-07-12

## Decisao refinada para implementacao

Este plano foi refinado por
[`plan_screening_agent_video_integration_refined.md`](concepts_video/plan_screening_agent_video_integration_refined.md).
As regras abaixo substituem os pontos anteriores que pediam extracao clinica
estruturada imediatamente apos qualquer upload de video.

- Manter `StateGraph` explicito; nao migrar para ReAct nesta etapa.
- Manter Chainlit como borda de I/O: mensagem, upload, timeout/cancelamento,
  streaming e progresso visual.
- O grafo decide quando precisa de video, quando deve apenas interpretar video e
  quando deve extrair contexto clinico estruturado.
- Upload direto ou `video_path=...` processa o video imediatamente pelo
  pipeline real.
- Pedido de video sem arquivo/path usa confirmacao em dois turnos: o grafo pede
  confirmacao, a confirmacao ativa `AskFileMessage`, e o upload reinvoca o
  grafo com o pedido pendente.
- Upload geral executa `video_analysis -> video_interpretation -> final_answer`.
- A interpretacao narrativa de video pode usar paciente ativo como contexto
  opcional, mas nao exige paciente.
- `video_clinical_context_json` e produzido apenas no fluxo de sintomas com
  video, antes de `symptom_analysis`.
- Perguntas gerais posteriores sobre o mesmo video reutilizam o artefato e nao
  acionam extracao clinica.
- `cl.Step` deve ser instrumentado em `app_chainlit.py` a partir dos eventos do
  `graph.astream(...)`, sem imports ou chamadas de Chainlit dentro dos nodes.
- `final_answer` continua sendo o unico node responsavel pela resposta final
  visivel ao usuario; se ele falhar, a resposta deve ser um erro tecnico seguro,
  nao uma resposta clinica fabricada.

Este documento continua o plano base em
[`plan_screening_agent_video_integration.md`](concepts_video/plan_screening_agent_video_integration.md)
e registra os ajustes de arquitetura e roteamento que vieram depois da primeira
versao do integrador de video.

## 0. Objetivo

Precisamos revisar e ajustar o fluxo de resposta, processamento de vídeo e roteamento entre os especialistas de vídeo e sintomas.

Inicie o planejamento a partir de:

1. Inspecione o comportamento atual no código e identifique a origem dos roteamentos incorretos e da abertura indevida da caixa de carregamento.
2. Use `langgraph_router_specialists_video_v2.ipynb` como referência funcional para a análise clínica integrada do vídeo.
3. Consulte a documentação oficial atual do LangGraph.
4. Avalie se uma arquitetura ReAct oferece ganho real frente ao grafo atual com estado e rotas condicionais explícitas. Documente benefícios, impacto de complexidade, riscos e recomendação antes de migrar a arquitetura. Não adote ReAct automaticamente.

## 1. Centralizar a resposta textual no Answer Node

Remova o disclaimer padrão fixo em inglês.

Toda mensagem textual apresentada ao usuário — inclusive mensagens atualmente produzidas por heurísticas, templates ou rotas de fallback — deve ser composta pelo LLM no `answer node`.

As regras determinísticas podem continuar controlando transições, validações e execução interna, mas não devem escrever diretamente mensagens conversacionais ao usuário.

O `answer node` deve receber o contexto necessário para compor a resposta final: pergunta do usuário, dados do paciente, análises dos especialistas, resultados do pipeline e qualquer orientação de segurança aplicável.

## 2. Executar e exibir análise clínica inicial após o upload do vídeo

Quando um vídeo for carregado, o sistema deve:

1. Processá-lo pelo pipeline de vídeo.
2. Reunir:
   - ficha e contexto clínico do paciente;
   - transcrição da fala;
   - evidências estruturadas do pipeline;
   - detecções visuais, comportamentais, posturais e emocionais relevantes.
3. Enviar esse contexto ao especialista em vídeo.
4. Gerar uma análise integrada do estado do paciente.
5. Exibir essa análise imediatamente ao usuário, por meio do `answer node`.
6. Persistir os resultados relevantes no estado para consultas posteriores.

A análise não deve apenas listar detecções. O especialista em vídeo deve relacionar o histórico clínico, a fala e as evidências observadas para produzir uma visão contextualizada da situação do paciente, seguindo a abordagem do notebook `langgraph_router_specialists_video_v2.ipynb`.

## 3. Criar contexto clínico estruturado derivado do vídeo

Adicione um campo novo e opcional ao estado para registrar a saída clínica estruturada do especialista em vídeo.

Esse campo deve conter, no mínimo:

- sintomas relatados na fala;
- sinais e observações relevantes visíveis no vídeo;
- evidências estruturadas que sustentam cada observação;
- limitações e incertezas da análise;
- possíveis pontos que merecem avaliação clínica mais estrita.

Esse contexto não substitui as saídas brutas do pipeline: ele as organiza para consumo clínico.

Quando a pergunta exigir análise de sintomas, condições ou possível agravamento com base no vídeo, o especialista de sintomas deve receber:

- o prompt do usuário;
- o contexto clínico do paciente;
- o contexto clínico estruturado produzido pelo especialista em vídeo;
- as evidências estruturadas relevantes do vídeo.

O especialista de sintomas não deve ser acionado para interpretar diretamente o vídeo bruto.

## 4. Corrigir o roteamento entre vídeo e sintomas

O roteador não deve encaminhar diretamente ao especialista de sintomas uma pergunta que dependa de análise audiovisual, mesmo quando a pergunta usar termos como “agravamento”, “condição” ou “sintoma”.

Por exemplo, a pergunta:

> “É possível detectar, com base no vídeo, o agravamento de alguma condição?”

deve seguir este fluxo:

`usuário → especialista em vídeo → especialista de sintomas → answer node`

Nesse fluxo:

1. O especialista em vídeo analisa ou reutiliza a análise já disponível do vídeo.
2. Ele extrai e estrutura sintomas, sinais e observações relevantes.
3. O especialista de sintomas interpreta essas informações junto ao contexto do paciente e à pergunta do usuário.
4. O `answer node` compõe a resposta final.

Os fluxos esperados são:

| Tipo de solicitação | Fluxo |
|---|---|
| Pergunta exclusivamente sobre sintomas, sem depender de vídeo | `usuário → especialista de sintomas → answer node` |
| Pergunta geral sobre vídeo | `usuário → especialista em vídeo → answer node` |
| Pergunta clínica baseada no vídeo | `usuário → especialista em vídeo → especialista de sintomas → answer node` |
| Upload de vídeo | `pipeline → especialista em vídeo → answer node` |

Perguntas estritamente sobre vídeo não devem passar pelo especialista de sintomas sem necessidade explícita. Exemplos: resumo do vídeo, conteúdo da fala, postura, expressões, comportamento e estado geral aparente do paciente.

## 5. Remover heurística que abre o carregamento ao citar “vídeo”

Atualmente, citar “vídeo” na mensagem parece abrir automaticamente a caixa de carregamento. Esse comportamento deve ser removido.

A menção a vídeo não deve, por si só, acionar a interface ou o fluxo de upload.

O LLM deve resolver a referência ao vídeo usando esta ordem:

1. Verificar se há um vídeo já carregado e processado no estado.
2. Confirmar com o usuário, quando houver ambiguidade, qual vídeo carregado deve ser usado.
3. Aceitar um caminho de vídeo informado pelo usuário, se aplicável.
4. Solicitar o envio ou o caminho de um vídeo somente quando não existir vídeo utilizável no estado.

A caixa de carregamento deve ser aberta apenas quando o usuário confirmar que deseja fornecer um novo vídeo, ou quando não houver vídeo disponível e ele precisar ser enviado para atender à solicitação.

## 6. Critérios de aceite

A implementação estará correta quando:

1. Não houver disclaimer fixo em inglês nem respostas conversacionais fixas geradas por heurísticas.
2. O `answer node` produzir todas as mensagens textuais finais ao usuário.
3. O upload processar o vídeo, gerar análise integrada com o contexto clínico, salvá-la no estado e exibí-la imediatamente.
4. O estado mantiver um campo estruturado de sintomas, sinais, evidências, limitações e observações clínicas derivadas do vídeo.
5. Perguntas gerais sobre vídeo forem tratadas apenas pelo especialista em vídeo.
6. Perguntas clínicas baseadas em vídeo passarem primeiro pelo especialista em vídeo e, quando necessário, pelo especialista de sintomas.
7. Perguntas exclusivamente sobre sintomas continuarem indo diretamente ao especialista de sintomas.
8. Mencionar “vídeo” não abrir a caixa de carregamento automaticamente.
9. Quando houver vídeo já processado, o LLM conseguir reutilizá-lo ou pedir confirmação ao usuário.
10. Houver testes cobrindo upload, reutilização de vídeo já processado, vídeo → sintomas, vídeo sem sintomas e sintomas sem vídeo.

## 7. Posicao sobre `ReAct`

Antes de alterar o roteador para uma estrutura `ReAct`, vale avaliar:

- se o ganho em clareza de decisao compensa o aumento de complexidade;
- se o problema atual pode ser resolvido com um roteador LLM mais explicito;
- se a orquestracao multipasso realmente precisa de planejamento dinamico ou
  apenas de um grafo melhor separado por intents.

Por enquanto, `ReAct` fica como opcao de investigacao, nao como decisao
imediata.

## 8. Proximo passo esperado

Usar este documento como base para plano detalhado de implementacao no grafo, com ajustes nos nodes de video, symptoms e answer, e com revisao do roteador antes de mexer em qualquer estrategia mais pesada como `ReAct`.
