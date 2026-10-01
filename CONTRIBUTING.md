# Contributing

Issues and pull requests are welcome. For a bug report, include the Mac model, macOS version, model checkpoint, command used, and the relevant error or behavior. Do not include private prompts, model files, credentials, or personal paths.

For a change, use a focused branch and pull request against this repository. Describe the behavior that changes and why. Keep the diff limited to the task, and avoid generated build output, downloaded model data, and machine-specific artifacts.

For code changes, run the relevant checks before opening a pull request. The model-free suite is:

~~~sh
make check-source check-native-cpu check-python-engine
~~~

This requires Xcode's Metal tools and the development dependencies. Model-backed tests need an explicit model and GPU run; report which checks were run and any that were skipped. For documentation-only changes, review rendered Markdown and run git diff --check.

See [DEVELOPMENT.md](DEVELOPMENT.md) for build and package details. A package or release is not implied by merging a pull request.
