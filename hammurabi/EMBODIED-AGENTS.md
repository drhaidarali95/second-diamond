# Hammurabi embodied agents — product specification

## Product goal

25 distinct interactive human-like 3D colleagues, with complete bodies, personal wardrobes, animation, expressive faces, voices, conversational understanding, memory, individual humor and genuine tool-backed task execution. Static portraits and descriptive profiles do not satisfy this requirement.

## First acceptance milestone: Atlas

Deliver one complete vertical slice before scaling assets to 25:

- Licensed, detailed adult human 3D model with full body, skin materials, hair, eyes and teeth; rigged skeleton, facial blendshapes and reusable animation rig. Preserve asset licenses and provenance.
- Navy professional wardrobe, plus business-casual variant; garments fitted to the model. Consistent physical identity across outfits.
- Interactive 3D scene with orbit/zoom, proper lighting, idle animation and grounded movement. Display real runtime state.
- Conversational audio input, interruption handling, audible replies, captions and synchronized mouth animation. Microphone permission and text fallback required.
- Stable licensed synthetic voice, evaluated for intelligibility and consistency. Human-authorized voice selection; no unauthorized real-person imitation.
- Connected language model using Atlas's role and authority policy. Personality: calm, decisive, candid; understated situational humor, no jokes during incidents or serious customer concerns. Style never overrides accuracy.
- Persistent conversation history and approved business memory, separated by project and user. Clear deletion, retention and provenance controls.
- One meaningful authorized task completed through an actual tool, with visible evidence, failure reporting and review. No generated text masquerading as execution.
- Founder-facing interface displays character, conversation, current task, tool activity and outcomes.

## Architecture

A browser-based 3D client loads production GLB assets with skeletal animation and facial morph targets. A server mediates model, speech and action providers; credentials stay server-side. The orchestration service persists tasks and audit records. Voice events drive lip-sync; model/tool events drive status and expression. Twenty-five identities share runtime code while retaining separate assets, voice bindings, personality prompts and scoped memory. Asset loading is on demand; render only visible characters.

Use appropriate licensed commercial character creation/export tooling or commissioned assets for believable humans. Blender is available for inspection, fitting and animation preparation. No production character assets have been acquired or generated. No 3D body, speech connection, animation or live model integration is currently complete.

## Dependencies requiring actual provisioning

1. Production-quality, commercially usable rigged human assets and wardrobe, including facial animation rights and compatible export format.
2. A funded conversational model provider with permitted network access; COMPANY_API_KEY and COMPANY_MODEL are absent.
3. A licensed speech provider or approved local speech models and suitable compute. No speech binding is configured.
4. Hosting accessible to the founder, HTTPS for microphone access, and authenticated access. A workspace file is not a deployed application.
5. A concrete first task and scoped tool connection for measuring useful execution.

Do not request credentials in chat. Use secure environment configuration when provisioning providers. Establish budgets before paid recurring operation. No public deployment, account purchase or model/asset procurement has occurred.

## Scale to all 25

After Atlas passes the complete interaction and task test, create and review the remaining identities using the same asset/animation pipeline. Each gets distinct body/face, outfits, voice, gesture style, humor rules and working behavior. Test consistency, accessibility, latency, interruption handling and task quality individually. Human-like presentation must remain clearly identified as AI and must not imply consciousness or unimplemented capabilities.
