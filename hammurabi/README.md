# Hammurabi — 25-agent company

Separate new project; the GCP bridge is untouched. Requires Python 3.11+; no third-party packages.

## Run the local pilot

```sh
cd /workspace/second-diamond/hammurabi
python3 company.py validate
python3 company.py demo
python3 company.py status
python3 -m unittest discover -s tests -v
```

Demo mode uses explicitly labeled fixed outputs to exercise orchestration and persistence; it is not autonomous AI work. Reports appear under reports/<run-id>/brief.md, with persistent tasks and audit events in data/company.sqlite3. Five ordered stages: Nova proposes, Orbit assigns, Relay designs, Keen reviews, Atlas summarizes. Runs stop awaiting founder review; they do not approve or execute business actions. Each invocation creates a new run. Repeated runs intentionally do not deduplicate business goals. Do not run unattended recurring jobs yet.

## Live inference

Configure COMPANY_API_KEY securely in the process environment and COMPANY_MODEL with an available Chat Completions-compatible OpenAI model. Do not put keys in commands, source files or chat. Allow api.openai.com through environment networking; this destination is not currently configured. The CLI sends the goal, each agent's prompt and earlier outputs to that provider. Use non-sensitive briefs until data handling is agreed.

```sh
python3 company.py live --goal 'Design a validation experiment for our first AI business'
```

No API key or model was supplied during implementation. Live inference is unverified. TLS verification remains enabled. Calls are bounded to five per run; configure provider spending controls. Failed calls halt downstream stages and are not automatically retried. Running the command again starts a new run and can incur new charges. Never silently substitute demo results for a failed live run.

## What is implemented

- CHARTER.md: ownership, departments, conduct, authority, coordination and training rules.
- agents/: 25 individual JSON profiles, prompts, personalities, missions, metrics and permissions.
- company.json: five-agent pilot selection and usage limits.
- company.py: sequential workflow, optional inference adapter, SQLite tasks/events and markdown reporting.
- tests/: offline integration and failure-path checks.

## Next activation steps

Choose the first business and measurable outcome; configure the model provider and budget; run and evaluate one live planning cycle. Then connect research and task tools with scoped permissions. Before 24/7 deployment, add durable hosting, scheduler, task leases, alerts, retention policy and provider cost accounting. Add action adapters and authorization enforcement before granting any external business capabilities. The profile permissions are descriptive; actual pilot safety comes from the absence of action tools.

## Interactive boardroom

The new `boardroom/` application supports selecting 25 colleagues, local Ollama conversation routing, saved meeting history, device-local speech options and a Three.js GLB character loader. See `boardroom/README.md` for setup and verified limitations. No model or character assets are installed; Zoom, Lovable and GitHub publication are not connected. This is not a completed embodied-agent deployment.
