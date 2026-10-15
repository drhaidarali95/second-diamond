# Second Diamond / Hammurabi

This repository contains two separate projects:

- The existing chat application: `app.py`, `requirements.txt`, `templates/` and `static/`.
- [Hammurabi](hammurabi/README.md): the 25-agent company foundation and local boardroom.

The Hammurabi addition preserves the existing application and its dependency files.

## Hammurabi checks

```sh
cd hammurabi
python3 -m unittest discover -s tests -v
python3 company.py validate
```

See [boardroom setup and current limitations](hammurabi/boardroom/README.md). Live models, production human avatars, Zoom participation and Lovable deployment are not yet connected.
