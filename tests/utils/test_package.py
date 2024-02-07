import importlib
import subprocess
import sys

import pytest
import torch

from edspdf.utils.package import package


def test_blank_package(frozen_pipeline, tmp_path):
    # Missing metadata makes poetry fail due to missing author / description
    with pytest.raises(Exception):
        package(
            pipeline=frozen_pipeline,
            root_dir=tmp_path,
            name="test-model",
            metadata={},
            project_type="poetry",
        )

    frozen_pipeline.package(
        root_dir=tmp_path,
        name="test-model",
        metadata={
            "description": "A test model",
            "authors": "Test Author <test.author@mail.com>",
        },
        project_type="poetry",
        distributions=["wheel"],
    )
    assert (tmp_path / "dist").is_dir()
    assert (tmp_path / "build").is_dir()
    assert (tmp_path / "dist" / "test_model-0.1.0-py3-none-any.whl").is_file()
    assert not (tmp_path / "dist" / "test_model-0.1.0.tar.gz").is_file()
    assert (tmp_path / "build" / "test-model").is_dir()


@pytest.mark.parametrize("package_name", ["my-test-model", None])
def test_package_with_files(frozen_pipeline, tmp_path, package_name):
    frozen_pipeline.save(tmp_path / "model")

    ((tmp_path / "test_model").mkdir(parents=True))
    (tmp_path / "test_model" / "__init__.py").write_text(
        """\
print("Hello World!")
"""
    )
    (tmp_path / "pyproject.toml").write_text(
        """\
[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "test-model"
version = "0.0.0"
description = "A test model"
authors = ["Test Author <test.author@mail.com>"]

[tool.poetry.dependencies]
python = "^3.7"
torch = "^{}"
""".format(
            torch.__version__.split("+")[0]
        )
    )

    with pytest.raises(ValueError):
        package(
            pipeline=frozen_pipeline,
            root_dir=tmp_path,
            version="0.1.0",
            name=package_name,
            metadata={
                "description": "Wrong description",
                "authors": "Test Author <test.author@mail.com>",
            },
        )

    package(
        name=package_name,
        pipeline=tmp_path / "model",
        root_dir=tmp_path,
        check_dependencies=True,
        version="0.1.0",
        distributions=None,
        metadata={
            "description": "A test model",
            "authors": "Test Author <test.author@mail.com>",
        },
    )

    module_name = "test_model" if package_name is None else "my_test_model"

    assert (tmp_path / "dist").is_dir()
    assert (tmp_path / "dist" / f"{module_name}-0.1.0.tar.gz").is_file()
    assert (tmp_path / "dist" / f"{module_name}-0.1.0-py3-none-any.whl").is_file()
    assert (tmp_path / "pyproject.toml").is_file()

    # pip install the whl file
    print(
        subprocess.check_output(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                str(tmp_path / "dist" / f"{module_name}-0.1.0-py3-none-any.whl"),
                "--force-reinstall",
            ],
            stderr=subprocess.STDOUT,
        )
    )

    module = importlib.import_module(module_name)

    assert module.__version__ == "0.1.0"

    with open(module.__file__) as f:
        assert f.read() == (
            (
                """\
print("Hello World!")
"""
                if package_name is None
                else ""
            )
            + """
# -----------------------------------------
# This section was autogenerated by edspdf
# -----------------------------------------

import edspdf
from pathlib import Path
from typing import Optional, Dict, Any

__version__ = '0.1.0'

def load(
    overrides: Optional[Dict[str, Any]] = None,
    device: "torch.device" = "cpu"
) -> edspdf.Pipeline:
    artifacts_path = Path(__file__).parent / "artifacts"
    model = edspdf.load(artifacts_path, overrides=overrides, device=device)
    return model
"""
        )


@pytest.fixture(scope="session", autouse=True)
def clean_after():
    yield

    print(
        subprocess.check_output(
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "-y",
                "test-model",
            ],
            stderr=subprocess.STDOUT,
        )
    )

    print(
        subprocess.check_output(
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "-y",
                "my-test-model",
            ],
            stderr=subprocess.STDOUT,
        )
    )
