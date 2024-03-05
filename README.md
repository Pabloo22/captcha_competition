# captcha_competition
Code for the Kaggle captcha competition for the course Machine Learning II.

## Project structure
The project structure of the project is based on the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template.

Main details of the project structure:
- Notebooks are used for exploratory data analysis, prototyping, visualizations, and debugging.
- The `captcha_competition` directory contains the source code of the project.
- Experiments are run in the `scripts` directory.
- The `data` directory contains the raw and processed data.
- The `models` directory contains the trained models.
- Hyperparameters and configurations are stored in the `config` directory as `.yaml` files.

## Installation

1. Install [poetry](https://python-poetry.org/docs/):

```bash
pip install poetry==1.7.1
```

2. Create a virtual environment:

```bash
poetry shell
```

3. Install the dependencies:

```bash
poetry install
```

For adding new dependencies, use:

```bash
poetry add <package>
```

## Members (alphabetical order)

- Pablo Ariño
- Álvaro Laguna
- Carlota Medrano

## Notes

- When installing everything remember to cahnge the environment path to your own computer.
- In order to add new fuctions, they must be added to the init.py

