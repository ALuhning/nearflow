# Contributing to Nearflow

Thank you for your interest in contributing to **Nearflow** — a customized fork of [Langflow](https://github.com/logspace-ai/langflow) developed and maintained by [Vital Point AI](https://vitalpoint.ai).

We welcome contributions of all kinds — whether you're submitting bug fixes, building new NEAR AI integrations, improving infrastructure, writing documentation, or suggesting ideas for future development.

To contribute to this project, please follow the standard [fork and pull request workflow](https://docs.github.com/en/get-started/quickstart/contributing-to-projects).

---

## 📋 Reporting Bugs or Suggesting Improvements

Please use the [GitHub Issues page](https://github.com/ALuhning/nearflow/issues) to:

- Report bugs
- Suggest new features
- Request documentation improvements

We use GitHub labels to help categorize issues and make them easier to find. Check out the [label list](https://github.com/ALuhning/nearflow/labels) for the current tagging system.

Need help using or extending Nearflow? Post your questions to [GitHub Discussions](https://github.com/ALuhning/nearflow/discussions) so others can benefit too. We do not provide individual support via email.

### 🛠 Tips for Reporting Issues

- **Describe your issue clearly:** Include steps to reproduce, relevant code snippets, and any error messages or logs.
- **Collapse long logs or code** using `<details>` tags so the issue remains easy to read:
  
  ```markdown
  <details>
    <summary>Show log output</summary>

    Your multiline log or error here
  </details>
  ```

## Contributing code and documentation

You can develop Langflow locally and contribute to the Project!

See [DEVELOPMENT.md](DEVELOPMENT.md) for instructions on setting up and using a development environment.

## 🚀 Opening a Pull Request (PR)

Once you've written and manually tested your changes, follow these steps to submit a PR:

- Push your feature or fix branch to your fork
- Open a new pull request against the `near-model` branch of this repository:  
  [https://github.com/ALuhning/nearflow](https://github.com/ALuhning/nearflow)
- Use [semantic commit message conventions](https://www.conventionalcommits.org/en/v1.0.0/):
  - `feat: add new nearai integration`
  - `fix: correct agent memory issue`
  - `docs: improve contributing guide`
- Ensure your PR description clearly explains:
  - What the change does
  - Why it’s necessary
  - How it was tested
  - Which issue(s) it resolves, if applicable (e.g., `Closes #42`)

We value clear and descriptive PRs that help reviewers understand your contribution and context quickly.

---

## 💡 GitHub Actions and Deployment

This project uses [GitHub Actions](https://github.com/features/actions) to:

- Build a Docker image from the `near-model` branch
- Push it to GitHub Container Registry (GHCR)
- SSH into your production server
- Pull the latest image
- Restart the `langflow` service using Docker Compose

Secrets and environment variables are managed using GitHub’s [Actions secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets).

> See `.github/workflows/deploy.yml` for the full pipeline configuration.

---

## 🙏 Thanks

Whether you're helping by submitting issues, writing code, improving documentation, or testing features — thank you!

Together, we’re building a secure, scalable, and decentralized user-owned AI interface using Langflow + NEAR AI.

— Aaron Luhning & the Vital Point AI team

