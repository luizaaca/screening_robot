# screening_agent video integration

The user asked to implement the existing plan for integrating the notebook video
flow into `screening_agent`. Treat the durable design as:

- Chainlit accepts uploaded videos and explicit `video_path=...` / `path: ...`.
- LangGraph owns `video_analysis` and `video_qa` intents.
- `video_analysis` calls the real `src/video_pipeline.process_video` path by
  default and stores serializable JSON plus a compact summary in state.
- `video_qa` uses a dedicated `SCREENING_AGENT_VIDEO_ANALYST_*` backend and may
  include active patient context, but video QA is still allowed without a patient.
- Fail closed on missing optional dependencies or pipeline errors; never invent a
  plausible analysis artifact and never reuse stale video output for a new video.

