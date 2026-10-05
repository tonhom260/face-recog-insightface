# Sync code and deploy

GitHub is the shared source of code. Each machine has its own checkout; edits in the cloud become available elsewhere after a commit and push. An uncommitted cloud working tree is not downloaded by `git fetch` or `git pull`.

## Development and review

Work on a feature branch such as `codex/mobile-attendance`. Commit tested source changes, then push that branch. Review the change in a pull request and merge into `main` when the release is ready. Track source, requirements, the npm lockfile, and model-download scripts; keep `.env`, model binaries, database files, and enrollments outside Git.

## Deployment machine

Use a separate deployment checkout on `main` with no local source edits. Initial installation and credentials are described in [development.md](development.md). When deploying an approved update:

```bash
git switch main
git pull --ff-only origin main
python3 scripts/prepare_models.py
docker compose up --build -d --wait
```

The model preparation command requires Python 3.11 or newer and Git LFS. It verifies pinned assets and skips existing valid downloads. Compose reads the deployment machine's ignored `.env`; pulling code does not supply passwords or synchronize the database. The commands above use the CPU container deployment. For native macOS/CoreML, pull the same approved code, install changed Python dependencies, rebuild `frontend`, then restart the backend service as described in [development.md](development.md).

`git fetch origin` downloads commits and updates remote branch references, without changing the checked-out files. `git pull --ff-only origin main` fetches and updates the current `main` checkout; it stops if a fast-forward is impossible. Run it when adopting a new release, not continuously. If it reports local changes or diverging commits, inspect them before proceeding; do not discard them blindly.

For deployment version records, capture `git rev-parse HEAD`. A later CI/CD workflow can build images tagged with that commit SHA, letting the deployment machine pull a chosen image version instead of building there.
