# Spheremix - Spherical Probability Distributions and Mixture Models

Spheremix is a library for spherical probability distributions and mixture models.

![ci](https://github.com/markvilar/spheremix/actions/workflows/ubuntu.yml/badge.svg)


### Getting started

#### Installing `uv`

```shell
# Install poetry
pip3 install --user uv
```

#### Building the project

```shell
# Build the project
uv build
```

#### Running scripts

```shell
uv run <script>
```

#### Invoking tools

```shell
# Invoke ruff
uv run ruff check .

# Invoke black
uv run black .

# Invoke pytest
uv run pytest
```


### Other uses

#### Managing packages

```shell
# Add a new package to the project
uv add <package>

# Remove a package from the project
uv remove <package>
```


### References

1) [https://docs.astral.sh/uv/getting-started/](https://docs.astral.sh/uv/getting-started/)
