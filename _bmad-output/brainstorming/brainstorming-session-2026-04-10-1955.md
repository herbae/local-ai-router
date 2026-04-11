---
stepsCompleted: [1, 2, 3, 4]
inputDocuments: [bootstrap.md]
session_topic: 'AI Router spike — local Ollama/Claude routing via OpenAI-compatible API'
session_goals: 'Validate feasibility on current hardware, identify best approach for rapid first spike, produce implementation plan to start building today'
selected_approach: 'ai-recommended'
techniques_used: ['first-principles-thinking', 'constraint-mapping', 'reverse-brainstorming']
ideas_generated: [Feasibility #1, Feasibility #2, Feasibility #3, Feasibility #4, Constraint #1, Constraint #2, Constraint #3, Constraint #4, Constraint #5, Risk #1, Risk #2, Risk #3, Risk #4, Risk Mitigation #1]
session_active: false
workflow_completed: true
---

# Brainstorming Session Results

**Facilitator:** iury
**Date:** 2026-04-10

## Session Overview

**Topic:** End-to-end AI Router spike — FastAPI service routing between Ollama (Gemma 4 E4B) and Claude Sonnet via OpenAI-compatible API
**Goals:** Validate feasibility on hardware (i7-11800H, 64GB RAM, RTX 3060 6GB), determine best approach for rapid first spike, produce actionable implementation plan for same-day start

### Context Guidance

_Source: bootstrap.md — Full project spec covering architecture (FastAPI + Ollama + Claude), routing criteria (token count, complexity, code detection), technical requirements (streaming, Docker Compose, config), and deliverables (router code, docker-compose, README, bootstrap script)._

### Session Setup

_Spike-oriented session. User wants speed over polish, working prototype over perfect design. "Come back and improve" iteration model. Goal is a testable router today._

## Technique Selection

**Approach:** AI-Recommended Techniques
**Analysis Context:** Rapid spike feasibility and architecture decisions

**Recommended Techniques:**

- **First Principles Thinking:** Strip the bootstrap.md spec to minimum viable spike — separate must-haves from nice-to-haves
- **Constraint Mapping:** Map hardware, time, knowledge, and complexity constraints to lock architecture decisions
- **Reverse Brainstorming:** Identify failure modes before they bite — find traps in API translation, Docker GPU, and classifier logic

## Technique Execution Results

### First Principles Thinking

**Interactive Focus:** What actually matters for a day-1 spike vs. the full product spec

**Key Breakthroughs:**

- **[Feasibility #1]**: Hardware Validation First — Before writing any router code, validate that Gemma 4 E4B runs on 6GB VRAM with acceptable latency. Everything else is wasted effort if local inference is too slow.
- **[Feasibility #2]**: Acceptable Latency Threshold — Simple factual queries must complete in <5 seconds end-to-end. Concrete benchmark: "What is the capital of Albania?"
- **[Feasibility #3]**: Zero Local Inference Experience — User has never run Ollama. Changes the spike scope from "build a router" to "discover if local LLMs are viable for me at all." Two-phase spike structure emerges.
- **[Feasibility #4]**: Model Flexibility — Gemma 4 E4B is first choice (hype-driven, valid for a spike), fallback to smaller models (Gemma 3 4B, Phi-4-mini). Router design must be model-agnostic.

**Core Insight:** The spike is actually two spikes — Spike 0 validates the model/hardware, Spike 1 builds the router. Spike 0 is a pass/fail gate.

### Constraint Mapping

**Interactive Focus:** Map every real constraint to force architecture decisions

**Key Findings:**

- **[Constraint #1]**: Strong Technical Foundation — Python, FastAPI, Docker all comfort zone. Only Ollama is new territory. No learning curve tax on core stack.
- **[Constraint #2]**: UI Goal — Open WebUI Chat Interface — Ideal outcome is Open WebUI with auto-routing. Open WebUI speaks OpenAI-compatible API, so if the router exposes `/v1/chat/completions` correctly, the UI is basically free.
- **[Constraint #3]**: API Access Solved — Claude Max subscription, no billing/credits constraint. Routing decision is about speed and quality, not cost savings.
- **[Constraint #4]**: Simple Heuristic Classifier for Day 1 — Route based on prompt length, code block detection, keyword patterns. Manual /local and /cloud overrides. No ML, no tiktoken.
- **[Constraint #5]**: No Streaming for Day 1 — Return complete responses only. Eliminates the hardest part of the router (streaming proxy with two different API formats). Massive complexity cut.

**Locked Constraint Table:**

| What | Decision | Status |
|---|---|---|
| Model | Gemma 4 E4B, fallback to smaller | Flexible |
| VRAM | 6GB hard limit | Fixed |
| Stack | Python, FastAPI, Docker — comfort zone | No risk |
| Ollama | Never used, needs validation first | Spike 0 |
| Classifier | Simple heuristics + manual override | Locked |
| UI | Open WebUI via Docker Compose | Day 1 goal |
| API key | Claude Max, user will provide | Solved |
| Cost concern | None — routing for speed/quality not savings | Clarified |
| Streaming | No — complete responses only | Locked |

### Reverse Brainstorming

**Interactive Focus:** "How could this spike fail?" — identify traps before hitting them

**Failure Modes Identified:**

- **[Risk #1]**: API Format Translation is the Core Engineering Challenge — Router sits between three API dialects. Open WebUI and Ollama speak OpenAI format, Claude speaks Anthropic format. This is where bugs will live.
- **[Risk Mitigation #1]**: ChatGPT as Zero-Translation Fallback — If Anthropic format translation is too painful, ChatGPT API speaks OpenAI format natively. Zero translation needed. Can swap to Claude later.
- **[Risk #2]**: NVIDIA Container Toolkit is a Hard Gate — GPU passthrough in Docker is non-negotiable. Without it, Ollama runs on CPU and the spike is useless. Must be part of Spike 0.
- **[Risk #3]**: Docker-First for Ollama, Native as Fallback — Try Ollama in Docker with GPU passthrough first (well-documented on Ubuntu). Only fall back to native install if Docker GPU fails.
- **[Risk #4]**: Imperfect Routing is Acceptable — Classifier will make wrong calls. Manual overrides are the safety net. "Good enough" + escape hatch = fine for day 1.
- **Failure Mode #7**: Pin Open WebUI version in docker-compose to avoid compatibility surprises.
- **Failure Mode #8**: Classify on the last message in the conversation history, not the whole array. Forward full history to Claude for context.
- **Failure Mode #9**: Anthropic API handles system prompts differently (separate `system` field). Router must extract and translate.

## Idea Organization and Prioritization

### Thematic Organization

**Theme 1: Hardware & Model Viability** — Feasibility #1-4, Risk #2
**Theme 2: Architecture & Scope Decisions** — Constraint #1-5, Risk #3-4
**Theme 3: API Translation & Integration Risks** — Risk #1, Risk Mitigation #1, Constraint #3, Failure Modes #7-9

### Prioritized Action Plan

#### Spike 0 — Hardware Validation (30 min)

1. Ensure nvidia-container-toolkit is installed on Ubuntu
2. Run Ollama container with GPU access (`--gpus all`)
3. Pull Gemma 4 E4B model
4. Test: "What is the capital of Albania?" — must complete <5 sec
5. **Pass -> proceed to Spike 1. Fail -> try smaller model or native Ollama fallback.**

#### Spike 1 — Build the Router (rest of day)

1. Project scaffolding — FastAPI app, config.yaml, .env
2. Simple classifier — prompt length + code detection + /local /cloud overrides
3. Ollama client — forward to Ollama's OpenAI-compatible endpoint (same format, easy)
4. Claude client — Anthropic API call + response translation to OpenAI format (hard part)
5. `/v1/chat/completions` endpoint — ties classifier + clients together
6. Docker Compose — router + ollama + open-webui, all wired up
7. Test through Open WebUI

#### Deferred to Iteration (not today)

- Streaming SSE support for both backends
- Sophisticated classifier (tiktoken, multi-step reasoning detection)
- README with setup instructions
- Bootstrap script for model pulling
- Config YAML for classifier thresholds
- Cost/usage tracking and logging improvements

## Session Summary and Insights

**Key Achievements:**

- Identified that the riskiest assumption (hardware viability) must be tested before any code
- Established a concrete, testable acceptance criterion (<5 sec for simple queries)
- Cut scope dramatically: no streaming, simple classifier, no documentation — just a working prototype
- Mapped all failure modes in advance — API translation, GPU passthrough, Open WebUI compatibility
- Discovered ChatGPT API as a zero-effort fallback if Claude translation proves painful
- Produced a clear two-phase spike with a pass/fail gate between phases

**Session Reflections:**

This was a focused, pragmatic session that avoided the trap of over-designing. The user's "test something quickly and come back to improve" mindset drove every decision toward the simplest viable approach. The two-phase spike structure (validate hardware THEN build) is the key insight — it prevents wasting a day building around a model that doesn't work on the hardware.
