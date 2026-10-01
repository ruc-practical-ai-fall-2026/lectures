# Fall 2024 Class 03 - Software Tools and Practice

In this class we review common tools we will use throughout the semester.

## Display of Presentations

Navigate to the `presentations` folder.

```bash
cd presentations
```

Start an http-server.

```bash
bash ../scripts/start_server.sh
```

## Installation

### Installation and Use via Github Codespaces (Only Recommended for Low-Memory Projects)

To use this repository via codespaces simply click on the `code` &rarr; `codespaces` &rarr; `create codespace on main` buttons.

Once the codespace is open in the browser, click the three bars in the top left corner and select `Open in VS Code Desktop`.

Widgets might work better when using VS Code Desktop vs. in the browser.

If the codespace takes a long time to build, use `Cmd` / `Ctrl` + `Shift` + `P` &rarr; `Codespaces: View Creation Logs` to check status.

If required, use `Cmd` / `Ctrl` + `Shift` + `P` &rarr; `Codespaces: Rebuild Container` to rebuild the container. Do not use `gh codespace rebuild`. This takes a long time since it re-downloads the entire image.

### Dependencies for Local Installation (Recommended for PyTorch Projects)

#### Poetry

This project is built on Python 3.12. Poetry is required for installation. To install Poetry, view the instructions [here](https://python-poetry.org/docs/).

In codespaces, Poetry installation is handled in the development container. The user does not need to install Poetry if working in codespaces.

### Local Installation

To install locally, first install the required dependencies, then clone the repository and navigate to its directory.

```bash
git clone <REPOSITORY NAME>
cd <REPOSITORY NAME>
```

Be sure to replace `<REPOSITORY NAME>` with the link to the repository that GitHub provides.

#### Installing Python Dependencies Locally

To install locally, first install the required dependencies (Poetry), then clone the repository and navigate to its directory.

Configure Poetry to install its virtual environment inside the repository directory.

```bash
poetry config virtualenvs.in-project true
```

Install the repository's Python dependencies.

If you only wish to install a virtual environment to use in Jupyter notebooks (no custom modules are needed) the use the following.

```bash
poetry install --no-root
```

If there are custom modules you need to install, then simply use `poetry install` without `--no-root`.

```bash
poetry install
```

Check where Poetry built the virtual environment with the following command.

```bash
poetry env info --path
```

Open the command pallette with `Ctrl` + `Shift` + `P` and type `Python: Select Interpreter`.

Now specify that VSCode should use the that interpreter (the one in `./.venv/Scripts/python.exe` for example, though this path will be system-specific). Once you specify this, Jupyter notebooks should show the project's interpreter as an option when you click the `kernel` icon or the small icon showing the current version of python (e.g., `Python 3.12.1`) and then click `Select Another Kernel`, and finally click `Python Environments...`.

### Troubleshooting

If you run into issues installing with Poetry, check which Python poetry is using.

```bash
poetry env info
```

If this is an older, unexpected, or incompatible version, you can change it with the following.

```bash
poetry env use "$PYTHON_EXE"
```

Where `$PYTHON_EXE` is some environment variable containing the path to your preferred Python. For example:

```bash
PYTHON_EXE=~/AppData/Local/Programs/Python/Python312/python.exe
```

Note that you must use a Python that is within the version range specified in the `pyproject.toml`!

## License

This repository is provided with an MIT license. See the `LICENSE` file.

## Contributing

Please email Mauro Sanchirico at ms3978@camden.rutgers.edu (academic) or sanchirico.mauro@gmail.com (personal) with questions, comments, bug reports, or suggestions for improvement.

