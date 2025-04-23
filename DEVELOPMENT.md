# Setting up a Development Environment

This document details how to set up a local development environment that will allow you to contribute changes to **Nearflow**, a secure, decentralized fork of [Langflow](https://github.com/logspace-ai/langflow), built and maintained by [Vital Point AI](https://vitalpoint.ai).

---

## Base Requirements

* The project is hosted on GitHub — you'll need a GitHub account.
* A code editor such as [Visual Studio Code](https://code.visualstudio.com/) is recommended.

---

## Set up Git Repository Fork

You will push changes to your own fork of the Nearflow repository, and from there create a Pull Request into the main Vital Point AI repository.

1. Fork the [Nearflow GitHub repository](https://github.com/ALuhning/nearflow/fork)

2. Clone your fork locally:

```bash
git clone https://github.com/<your-username>/nearflow.git
cd nearflow
```

3. Add the upstream repository:

```bash
git remote add upstream https://github.com/ALuhning/nearflow.git
git remote set-url --push upstream no_push
```

> **Windows/WSL Users**: If files appear modified due to file mode differences (`100755 → 100644`), use:  
> `git config core.filemode false`

---

## Set up Environment

You have two main setup options:

### Option 1 (Preferred): Use a Dev Container

Open this repository as a [Dev Container](https://containers.dev/) per your IDE’s instructions.

#### VS Code Instructions

* See [Developing inside a Container](https://code.visualstudio.com/docs/devcontainers/containers)
* You may want to [share Git credentials](https://code.visualstudio.com/remote/advancedcontainers/sharing-git-credentials) with the container

---

### Option 2: Use Your Own Local Environment

#### Install Pre-Requisites

* **Operating System**: macOS or Linux (Windows must use WSL)
* **`git`**: For version control
* **`make`**: For build automation
* **`uv`**: Python project manager (`>=0.4`) — [Install here](https://docs.astral.sh/uv/getting-started/installation/)
* **`npm`** and Node.js (`v22.x`): For frontend build — [Install via package manager](https://nodejs.org/en/download/package-manager)

> Windows users: make sure `npm` is installed inside WSL.

---

### Initial Environment Validation

Run the following from your project root:

```bash
make init
```

This initializes the backend and frontend, builds the static frontend, and starts the server. You should see:

```
🟢 Open Nearflow → http://127.0.0.1:7860
```

Visit the provided URL in your browser to confirm everything is working.

Stop the server with `CTRL+C` — you’re now ready for development mode.

---

## Optional Pre-Commit Hooks

To keep your commits clean and auto-formatted:

```bash
uv sync
uv run pre-commit install
```

> With this installed, run commits like:  
> `uv run git commit -m "feat: your change"`

---

## Run Nearflow in Development Mode

Running Nearflow in dev mode enables hot-reload for both backend and frontend code.

> You’ll typically have multiple terminals open for:  
> *Backend*, *Frontend*, *Docs*, and *Build* tasks

---

### Backend Service

Start the FastAPI backend in one terminal:

```bash
make backend
```

It will run on: [http://localhost:7860/health](http://localhost:7860/health)

---

### Frontend Service

Start the React frontend in another terminal:

```bash
make frontend
```

It will run on: [http://localhost:3000](http://localhost:3000)

> Use this for most interactive development.

---

### Docs (Optional)

To preview or contribute to documentation (Docusaurus):

```bash
cd docs
yarn install
yarn start
```

It will run on: [http://localhost:3001](http://localhost:3001)

---

## Adding or Modifying a Component

Components live under:
```
src/backend/base/langflow/components/
```

- **NEAR AI custom components** are primarily under:
  ```
  src/backend/base/langflow/components/langchain_utilities/
  ```

To add a component:
- Create the `.py` file
- Register it in the module’s `__init__.py`
- Restart the backend and refresh the frontend

✅ Components will hot-reload on save, but may require a browser refresh to appear in the UI.

Please also:

- Add or update a corresponding Markdown test file
- Add or update unit tests (in `tests/unit/components/`)

---

## Building and Testing Changes

Before committing:

```bash
make lint
make format_backend
make format_frontend
make unit_tests
```

To simulate a clean install:

```bash
make init
```

Visit [http://localhost:7860](http://localhost:7860) and test your changes in a clean session.

---

## Committing, Pushing, and Pull Requests

1. Create a new feature branch:

```bash
git checkout -b feat/your-feature-name
```

2. Commit your changes:

```bash
uv run git commit -m "feat: add new NEAR AI component"
```

3. Push to your fork and open a Pull Request against the `near-model` branch at:  
   [https://github.com/ALuhning/nearflow](https://github.com/ALuhning/nearflow)

---

## Some Quirks!

### Testing

Some tests may pass in `pytest` but fail in `make unit_tests`.  
You can test a file directly with:

```bash
uv run pytest src/backend/tests/unit/your_component_test.py
```

---

### Changing Files

- Changes to `.starter_projects` are expected on rebuild
- `uv.lock` and `package-lock.json` can change during builds — don't commit them unless needed

To ignore them in your local repo:

```bash
git update-index --assume-unchanged uv.lock src/frontend/package-lock.json
```

To re-enable:

```bash
git update-index --no-assume-unchanged uv.lock src/frontend/package-lock.json
```

---

## 🧠 Nearflow Tips

- Use the Langflow canvas to prototype and copy working logic back to `.py` files
- Develop components in isolation by wiring them up in a blank flow
- Watch logs carefully for issues with memory, agent threads, or NEAR API interactions

---

## 🙌 Final Notes

Thanks for contributing to **Nearflow**!

Whether you're fixing bugs, building NEAR AI components, or improving infrastructure — you’re helping build a secure, decentralized, user-owned, and enterprise-ready AI interface platform.

— Aaron Luhning, Vital Point AI
