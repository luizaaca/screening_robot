# Plano Ajustado: Integração de Vídeo no Screening Agent

## Resumo

Manter a arquitetura com `StateGraph` explícito, Chainlit apenas como borda de I/O e `final_answer` como único nó responsável por texto conversacional visível ao usuário.

A regra ajustada é: vídeo novo não dispara extração clínica por padrão. Após upload, o agente executa o pipeline real de vídeo e faz apenas a interpretação narrativa associada ao contexto do paciente, se esse contexto estiver carregado. A extração clínica estruturada do vídeo só será chamada quando o vídeo participar de um fluxo de análise de sintomas.

## Mudanças Principais

- Preservar o modelo de “um vídeo ativo” por thread.
- Remover heurísticas de upload em `app_chainlit.py`; a decisão de pedir vídeo deve vir do grafo.
- Usar confirmação em dois turnos para pedido de upload:
  - resposta do agente pergunta se o usuário quer enviar vídeo;
  - confirmação ativa `AskFileMessage`;
  - arquivo enviado reinvoca o grafo com o pedido pendente.
- Upload direto ou caminho local válido pula a confirmação e processa o vídeo imediatamente.
- Novo vídeo sempre executa `process_video(...)` real.
- Após processamento de vídeo novo:
  - fluxo geral de vídeo: chamar somente interpretação narrativa;
  - se houver contexto de paciente carregado, passá-lo para a interpretação;
  - se não houver contexto, interpretar apenas com base no artefato do vídeo;
  - não chamar extração clínica estruturada.
- Fluxo de sintomas com vídeo:
  - chamar extração clínica estruturada somente aqui;
  - passar o resultado estruturado para o nó de análise de sintomas.
- Reutilizar artefatos do vídeo ativo:
  - não rerodar pipeline para o mesmo vídeo;
  - não rerodar extração clínica se já existir para o mesmo vídeo e mesma solicitação clínica relevante;
  - permitir nova interpretação narrativa quando a pergunta do usuário mudar.

## Interfaces e Estado

Adicionar ou ajustar campos no estado do grafo:

- `pending_video_request`: pedido de vídeo aguardando confirmação/upload.
- `video_input_status`: `none`, `awaiting_confirmation`, `awaiting_upload`.
- `video_artifact_path`, `video_summary`, `video_json`, `video_status`, `video_error`: manter contrato atual do pipeline.
- `video_interpretation`: resposta narrativa do especialista de vídeo para perguntas gerais ou contextualizadas pelo paciente.
- `video_clinical_context_json`: extração clínica estruturada, preenchida apenas no fluxo de sintomas.
- `turn_outcome`: evento estruturado para o `final_answer` montar a resposta final sem texto fixo nos nós intermediários.

Definir um contrato enxuto para `video_clinical_context_json`, voltado ao consumo pelo analisador de sintomas:

- sintomas relatados ou inferidos a partir do pedido;
- sinais observáveis no vídeo;
- evidências com timestamps quando disponíveis;
- limitações do vídeo;
- incertezas relevantes;
- pontos de atenção clínica.

## Especificação de Progresso com `cl.Step`

Adicionar progresso visual com `cl.Step` para cada etapa principal de processamento executada pelo grafo durante uma interação no Chainlit. A instrumentação deve ficar na camada Chainlit, em `app_chainlit.py`, preferencialmente dentro de `_stream_graph_turn`, observando os eventos emitidos por `graph.astream(...)`.

Não inserir `cl.Step`, imports de Chainlit ou chamadas de UI dentro dos nodes do grafo. Os nodes devem continuar portáveis, testáveis e executáveis fora do Chainlit.

### Fonte dos eventos

- Usar o stream já existente de LangGraph como fonte de verdade para abertura, atualização e fechamento dos steps.
- Manter `subgraphs=True` e `version="v2"` para preservar visibilidade dos eventos do grafo.
- Se necessário, incluir um modo de stream adicional para eventos de progresso, desde que isso não substitua o streaming atual de tokens do `final_answer`.
- Não depender de logs de console ou de strings de resposta do usuário para inferir progresso.

### Granularidade padrão

Exibir steps apenas para nodes principais do fluxo:

- `router`;
- `patient_lookup`;
- `video_analysis`;
- `video_interpretation` ou `video_qa`, conforme o nome final adotado na implementação;
- `video_clinical_extraction`;
- `symptom_analysis`;
- `final_answer`;
- etapa final de erro, quando aplicável.

Não exibir por padrão nodes internos de subgrafos, tool calls internas ou etapas auxiliares de captura. Isso evita ruído na UI e reduz risco de exposição de detalhes clínicos ou técnicos.

### Conteúdo exibido

Cada `cl.Step` deve mostrar somente informação operacional resumida e sanitizada:

- nome humano curto;
- status em andamento, concluído, erro ou cancelado;
- resumo curto do que a etapa fez;
- erro técnico sanitizado quando a etapa falhar.

Não mostrar na UI padrão:

- prompt completo;
- histórico completo da conversa;
- payload clínico bruto;
- JSON completo do pipeline de vídeo;
- argumentos completos de tools;
- dados sensíveis de paciente além do que já estiver autorizado na resposta final.

Payloads técnicos detalhados devem continuar restritos ao modo de debug já existente, não ao `cl.Step` padrão.

### Ciclo de vida dos steps

- Abrir um `cl.Step` quando o evento indicar início de node principal.
- Fechar ou atualizar o step quando o evento indicar conclusão do mesmo node.
- Marcar `step.is_error = True` quando o node falhar.
- Usar `step.output` apenas para resumo sanitizado.
- Evitar steps duplicados para o mesmo node em um único turno, exceto quando o grafo executar legitimamente o mesmo node mais de uma vez.
- Preservar o streaming de tokens do `final_answer` para a mensagem final; o step de `final_answer` deve representar a etapa, não substituir a resposta.

### Nomes sugeridos para UI

- `router`: "Classificando solicitação"
- `patient_lookup`: "Buscando paciente"
- `video_analysis`: "Processando vídeo"
- `video_interpretation` ou `video_qa`: "Interpretando vídeo"
- `video_clinical_extraction`: "Extraindo contexto clínico do vídeo"
- `symptom_analysis`: "Analisando sintomas"
- `final_answer`: "Gerando resposta"
- fallback de erro: "Tratando erro"

### Critérios de aceite

- A UI mostra progresso para os nodes principais durante a execução.
- A UI não mostra nodes internos de subgrafos por padrão.
- A mensagem final continua sendo renderizada pelo fluxo atual de resposta.
- Falhas de node aparecem como step com erro e resposta técnica segura.
- Nenhum prompt completo, payload clínico bruto ou JSON completo de vídeo aparece em `cl.Step` por padrão.
- Os testes conseguem validar a criação dos steps usando stubs/fakes de Chainlit e do stream do grafo.

## Fluxo do Grafo

- Sintomas sem vídeo: `router -> symptom_analysis -> final_answer`.
- Pergunta geral sobre vídeo já ativo: `router -> video_interpretation -> final_answer`.
- Upload de vídeo sem pedido de sintomas: `router -> video_pipeline -> video_interpretation -> final_answer`.
- Pedido de sintomas com vídeo já ativo: `router -> video_clinical_extraction -> symptom_analysis -> final_answer`.
- Pedido de sintomas que requer upload: `router -> final_answer` pede confirmação; após upload, `video_pipeline -> video_clinical_extraction -> symptom_analysis -> final_answer`.
- Cancelamento, timeout ou erro técnico: registrar evento estruturado e deixar `final_answer` responder. Se o próprio `final_answer` falhar, retornar erro técnico explícito, sem fallback clínico fabricado.

## Documentação

Atualizar primeiro o plano de continuação em `concepts_video/plan_screening_agent_video_integration_continuation.md` com esta decisão refinada.

Na implementação posterior, atualizar também `README.md` e `README_pt-br.md` para refletir:

- Chainlit como camada de upload;
- grafo como dono da decisão;
- pipeline real de vídeo;
- interpretação narrativa por padrão após upload;
- extração clínica somente no fluxo de sintomas.

## Testes

Cobrir os cenários principais:

- Upload de vídeo geral executa `process_video(...)` e chama interpretação narrativa, sem chamar extração clínica.
- Upload com paciente carregado passa o contexto do paciente para a interpretação narrativa.
- Upload sem paciente carregado ainda gera interpretação narrativa baseada apenas no vídeo.
- Fluxo de sintomas com vídeo chama extração clínica estruturada e entrega o JSON ao analisador de sintomas.
- Fluxo de sintomas com vídeo já processado reutiliza o artefato do pipeline.
- Pergunta geral posterior sobre o mesmo vídeo não aciona extração clínica.
- Confirmação em dois turnos funciona para aceitar, recusar, timeout e envio direto.
- Textos visíveis continuam concentrados em `final_answer`.
- Falha do modelo de resposta final retorna erro técnico, sem resposta clínica determinística fabricada.

## Assumptions

- O backend do especialista de vídeo pode executar chamadas separadas para interpretação narrativa e extração clínica.
- Extração clínica é uma etapa lazy: só nasce quando o roteador identifica análise de sintomas com vídeo.
- O contexto do paciente é opcional para interpretação narrativa e obrigatório apenas quando o fluxo já tiver carregado esse contexto por outro caminho.
- A integração deve permanecer compatível com o notebook de referência, mas o runtime do agente deve priorizar os contratos atuais em `src/video_pipeline` e `screening_agent`.

## Checklist de Implementação para Agente Futuro

Use esta seção como checklist operacional. Marque cada item conforme for implementando.

### Preparação

- [x] Confirmar o estado atual de `app_chainlit.py`, `src/screening_agent/graph/builder.py`, `src/screening_agent/graph/state.py` e nodes existentes antes de editar.
- [x] Confirmar a API instalada de Chainlit para `cl.Step`, `AskFileMessage` e streaming.
- [x] Confirmar a API instalada de LangGraph para `graph.astream(...)`, eventos, `subgraphs=True` e `version="v2"`.
- [x] Identificar testes existentes que cobrem vídeo, Chainlit, builder e streaming.

### Estado e contratos do grafo

- [x] Adicionar ou ajustar campos de estado para `pending_video_request`, `video_input_status`, `video_interpretation`, `video_clinical_context_json` e `turn_outcome`.
- [x] Manter o contrato de um vídeo ativo por thread.
- [x] Garantir que artefatos do pipeline real de vídeo sejam reutilizados para o mesmo vídeo.
- [x] Garantir que `video_clinical_context_json` seja criado somente no fluxo de análise de sintomas.

### Roteamento e fluxo

- [x] Ajustar o router para distinguir pedido geral de vídeo, upload/caminho de vídeo, análise de sintomas com vídeo e confirmação/recusa de upload.
- [x] Implementar confirmação em dois turnos para pedidos de upload.
- [x] Permitir que upload direto ou caminho local válido pule a confirmação.
- [x] Garantir que upload geral execute `video_pipeline -> video_interpretation -> final_answer`.
- [x] Garantir que sintomas com vídeo executem `video_clinical_extraction -> symptom_analysis -> final_answer`.
- [x] Garantir que pergunta geral posterior sobre vídeo ativo não acione extração clínica.

### Nodes de vídeo

- [x] Separar pipeline real de vídeo, interpretação narrativa e extração clínica estruturada em responsabilidades claras.
- [x] Chamar `process_video(...)` real para vídeo novo.
- [x] Fazer interpretação narrativa após upload com contexto de paciente apenas quando houver paciente ativo.
- [x] Fazer extração clínica estruturada apenas quando o fluxo for análise de sintomas.
- [x] Propagar erros de vídeo como eventos estruturados para `final_answer`, sem resposta clínica fabricada.

### Chainlit e upload

- [x] Remover heurísticas de upload baseadas apenas em palavras-chave na camada Chainlit.
- [x] Manter Chainlit responsável apenas por mensagem, upload, timeout/cancelamento e chamada do grafo.
- [x] Reinvocar o grafo após upload, cancelamento ou timeout com estado/evento adequado.
- [x] Preservar validação de tipo e tamanho de vídeo já existente.

### `cl.Step`

- [x] Instrumentar `_stream_graph_turn` para criar `cl.Step` a partir de eventos do `graph.astream(...)`.
- [x] Criar steps apenas para nodes principais.
- [x] Usar nomes humanos curtos conforme a especificação.
- [x] Atualizar `step.output` apenas com resumo sanitizado.
- [x] Marcar `step.is_error = True` quando o node falhar.
- [x] Preservar streaming de tokens do `final_answer`.
- [x] Garantir que prompts, payload clínico bruto, JSON completo de vídeo e tool args não apareçam em steps padrão.

### Resposta final

- [x] Centralizar textos visíveis no `final_answer`.
- [x] Remover respostas conversacionais fixas dos nodes intermediários quando elas concorrerem com `final_answer`.
- [x] Garantir erro técnico explícito se o próprio `final_answer` falhar.

### Documentação

- [x] Atualizar `concepts_video/plan_screening_agent_video_integration_continuation.md` se ele ainda for usado como plano de continuidade.
- [x] Atualizar `README.md` com fluxo de vídeo, upload, interpretação, extração clínica lazy e progresso visual.
- [x] Atualizar `README_pt-br.md` com o mesmo conteúdo em português.

### Testes e verificação

- [x] Testar upload geral com pipeline real mockado/injetado e interpretação narrativa sem extração clínica.
- [x] Testar upload com paciente ativo passando contexto para interpretação narrativa.
- [x] Testar upload sem paciente ativo gerando interpretação baseada apenas no vídeo.
- [x] Testar análise de sintomas com vídeo chamando extração clínica estruturada.
- [x] Testar reutilização de vídeo ativo sem rerodar pipeline.
- [x] Testar pergunta geral posterior sem acionar extração clínica.
- [x] Testar confirmação, recusa, timeout e envio direto.
- [x] Testar criação de `cl.Step` para nodes principais.
- [x] Testar que nodes internos não aparecem por padrão.
- [x] Testar falha de node marcando step como erro.
- [x] Testar que conteúdo sensível ou payload bruto não aparece nos steps.
- [x] Executar os testes focados de Chainlit, grafo, builder e vídeo antes de encerrar.
