# Hammurabi Boardroom

A local web application for the 25-agent company. No Lovable generation calls, paid inference adapter, or subscription integration is used by this application. Existing hardware, electricity, hosting and platform subscriptions are not guaranteed free.

## Run

Requires Python 3.11+, Node/npm for installing Three.js, and a modern browser.

```sh
cd /workspace/second-diamond/hammurabi/boardroom
npm ci --ignore-scripts
python3 server.py
```

Open port 8765 on the machine running the server. This server deliberately binds to loopback and rejects other origins/hosts. It is not publicly deployed. The onboarding chat cannot display this running application to the founder. A localhost address is not a shareable preview.

## Local intelligence, without model API credits

Install Ollama on your own suitable machine, select a model whose license permits your intended use, download it, and run Ollama locally. This environment currently has neither Ollama nor a model. Set BOARDROOM_MODEL to the exact installed model name before starting server.py. The server connects exclusively to 127.0.0.1:11434 with no paid provider fallback. We have not verified live inference or measured model quality. Download/compute/storage requirements depend on the model and hardware.

For each agent, the server uses its role prompt and recent per-meeting conversation history. Selected colleagues respond sequentially; later colleagues receive earlier responses as context. Messages and replies persist in state.sqlite3; transcript export is available. Reports and spoken responses are conversation, not external business task execution. Action tools are still absent. Long meetings use only the most recent 20 messages per agent as context; there is no summarizing long-term memory engine yet.

## Speech

The UI lists only voices the browser marks localService. Speech output uses that device's built-in speech synthesis, not a hosted speech API. Voice quality and availability vary and are not validated in this headless environment. Local dictation requires browser processLocally support and an installed language pack; otherwise use text. The app intentionally does not fall back to hosted browser dictation. Production voice identity and synchronized lip motion are not implemented.

## Human characters

Place licensed GLB characters in assets/<agent-id>.glb (for example assets/atlas.glb). The local Three.js viewer loads the skeleton, materials and first embedded animation, with orbit controls. Character files should embed textures and resources; remote asset dependencies are unsupported. Large assets can exhaust browser memory when all 25 are selected. Each character needs independently reviewed commercial-use rights.

No character assets have been supplied. The application explicitly shows a missing-asset state; it does not substitute geometric figures or portraits. The GLB code path is implemented but not validated against a production character. Clothing changes, facial expression controls and lip synchronization still need implementation after choosing the rig/morph target format.

## Zoom: two distinct deliverables

1. A boardroom image/audio stream in a Zoom meeting: run the app on the founder's machine and use a compatible virtual camera and audio-routing setup. This requires end-to-end testing on that operating system. A composite feed appears as one Zoom participant, even when the frame shows 25 characters.
2. Twenty-five independent Zoom attendees: requires a supported meeting-client/bot architecture, suitable accounts or SDK authorization, audio/video routing, meeting permissions and platform-limit verification. No SDK, participant bot or Zoom account is connected here. Do not claim that an embedded meeting link creates AI attendees or that this has been verified to cost zero credits/fees.

The ultimate target remains individual agents that can attend real meetings and carry out verified work. This application is a runtime foundation, not completion of that target.

## GitHub and Lovable handoff

The founder selected drhaidarali95/second-diamond. The original application is preserved, and Hammurabi lives under hammurabi/ on a separate feature branch. Git read access is verified. Remote publication is reported separately after testing the push. No Lovable tool is available in this session. No Lovable project or deployment has been created. Do not request or paste credentials in chat.

Lovable AI generation may consume credits. Under the no-credit requirement, develop code directly and use a verified repository sync/import route only if the user's project supports it without generation credits. This availability is not yet verified. Static frontend hosting alone cannot run the Python service or local model; the final hosting architecture must account for that separation.

## Validation performed

- All 25 profiles loaded and appeared in a headless Chromium DOM.
- Missing model and missing assets are explicitly reported.
- Local Three.js module served successfully.
- Conversation persistence and failure behavior tested with an injected offline test provider.
- No real model reply, audio quality, human-character render, tool action or Zoom attendance has been claimed as verified.
