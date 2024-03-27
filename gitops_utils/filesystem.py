import json
import os
import os.path
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional

import git
import hcl2
import lark.exceptions
import requests
from filelock import FileLock
from filelock import Timeout as FileLockTimeout
from ruamel.yaml import YAML, StringIO, YAMLError

from gitops_utils import defaults
from gitops_utils.cases import is_nothing
from gitops_utils.exports import format_results
from gitops_utils.logs import Logs
from gitops_utils.matchers import is_url
from gitops_utils.transforms import Transforms
from gitops_utils.types import FilePath


def find_working_dir_for_repository_in_path(
    path: Optional[FilePath] = ".",
) -> Optional[str]:
    """
    Find the working directory for a repository in the given path.

    Args:
        path (Optional[FilePath]): The path to search for the repository. Defaults to ".".

    Returns:
        Optional[str]: The working directory of the repository if found, else None.
    """
    if path is None:
        path = "."

    try:
        repo = git.Repo(path, search_parent_directories=True)
        return repo.working_tree_dir
    except git.GitError:
        return None


def file_path_depth(file_path: FilePath) -> int:
    """
    Returns the depth of a file path.

    Args:
        file_path (str): The file path.

    Returns:
        int: The depth of the file path.
    """
    return 0 if is_nothing(file_path) else len(Path(file_path).parts)


def file_path_rel_to_root(file_path: FilePath):
    """
    Returns a normalized path relative to the root directory.

    Args:
        file_path (FilePath): The file path to calculate the relative path from.

    Returns:
        str: The normalized path relative to the root directory.
    """
    depth = file_path_depth(file_path)
    path_rel_to_root = [os.pardir for _ in range(depth)]
    return os.path.join(*path_rel_to_root)


class Filesystem(Logs, Transforms):
    def __init__(self, cur_dir: Optional[FilePath] = None, **log_opts):
        super().__init__(**log_opts)
        super().__init__()
        self.tld = find_working_dir_for_repository_in_path(cur_dir)
        self.tld = Path(self.tld) if self.tld else self.tld

    def get_file(
        self,
        file_path: FilePath,
        decode: Optional[bool] = True,
        return_path: Optional[bool] = False,
        charset: Optional[str] = "utf-8",
        errors: Optional[str] = "strict",
        headers: Optional[Mapping[str, str]] = None,
        raise_on_not_found: bool = False,
    ):
        if headers is None:
            headers = {}

        file_data = ""

        def state_negative_result(result: str):
            self.logger.warning(result)

            if raise_on_not_found:
                raise FileNotFoundError(result)

        try:
            if is_url(str(file_path)):
                self.logged_statement(f"Getting remote URL: {file_path}")
                response = requests.get(file_path, headers=headers)
                if response.ok:
                    file_data = response.content.decode(charset, errors)
                else:
                    state_negative_result(
                        f"URL {file_path} could not be read:"
                        f" [{response.status_code}] {response.reason}"
                    )
            else:
                local_file = self.local_path(file_path)

                self.logged_statement(f"Getting local file: {local_file}")

                if local_file.is_file():
                    file_data = local_file.read_text(charset, errors)
                else:
                    state_negative_result(f"{file_path} does not exist")
        except ValueError as exc:
            self.logger.warning(f"Reading {file_path} not supported: {exc}")
            decode = False

        if decode:
            self.logged_statement(f"Decoding {file_path}")
            file_data = (
                {}
                if is_nothing(file_data)
                else self.decode_file(file_data=file_data, file_path=file_path)
            )

        retval = [file_data]

        if return_path:
            retval.append(file_path)

        if len(retval) == 1:
            return retval[0]

        return tuple(retval)

    def decode_file(
        self,
        file_data: str,
        file_path: Optional[FilePath] = None,
        suffix: Optional[str] = None,
    ):
        yaml = YAML(typ="safe")
        file_data_stream = StringIO(file_data)

        if suffix is None:
            if file_path is not None:
                self.logger.info(f"Decoding file {file_path}")
                suffix = Path(file_path).suffix.lstrip(".").lower()

        if suffix == "yml" or suffix == "yaml":
            self.logger.info(f"Data is being loaded from YAML")

            try:
                return yaml.load(file_data_stream)
            except YAMLError as exc:
                self.logger.warning(
                    f"Decoding data from YAML resulted in error: {exc},"
                    f" trying to read it as multiple documents"
                )

                try:
                    yaml_data = {}

                    for doc in yaml.load_all(file_data_stream):
                        yaml_data = self.merger.merge(yaml_data, doc)

                    return yaml_data
                except YAMLError as exc:
                    raise RuntimeError(f"Failed to parse YAML file") from exc
        elif suffix == "json":
            self.logger.info(f"Data is being loaded from JSON")

            try:
                return json.loads(file_data)
            except (json.JSONDecodeError, TypeError) as exc:
                raise RuntimeError(f"Failed to parse JSON file") from exc
        elif suffix == "tf":
            self.logger.info(f"Data is being loaded from HCL2")

            try:
                return hcl2.load(file_data_stream)
            except lark.exceptions.ParseError as exc:
                raise RuntimeError(f"Failed to parse HCL file") from exc
        else:
            try:
                return json.loads(file_data)
            except json.JSONDecodeError as exc:
                self.logger.info(f"Failed to decode from JSON, trying YAML: {exc}")

                try:
                    return yaml.load(file_data_stream)
                except YAMLError as exc:
                    self.logger.warning(
                        f"Decoding data from YAML resulted in error: {exc},"
                        f" trying to read it as multiple documents"
                    )

                    try:
                        yaml_data = {}

                        for doc in yaml.load_all(file_data_stream):
                            yaml_data = self.merger.merge(yaml_data, doc)

                        return yaml_data
                    except YAMLError as exc:
                        self.logger.info(
                            f"Failed to decode from YAML, trying HCL: {exc}"
                        )
                        try:
                            return hcl2.load(file_data_stream)
                        except lark.exceptions.ParseError as exc:
                            raise RuntimeError(
                                f"Failed to parse raw data, format is unknown and all known parsers failed"
                            ) from exc

    def update_file(
        self,
        file_path: FilePath,
        file_data: Any,
        encode_with_json: Optional[bool] = False,
        allow_empty: Optional[bool] = False,
    ):
        if is_nothing(file_data) and not allow_empty:
            self.logger.warning(f"Empty file data for {file_path} not allowed")
            return None

        if encode_with_json:
            file_data = format_results(file_data, format_json=True)

        if not isinstance(file_data, str):
            file_data = str(file_data)

        self.logged_statement(f"Updating local file {file_path}", verbose=True)
        lock = FileLock(f"{file_path}.lock", timeout=defaults.MAX_FILE_LOCK_WAIT)

        try:
            with lock.acquire():
                local_file = self.local_path(file_path)

                self.logged_statement(f"Updating local file: {local_file}")

                local_file.parent.mkdir(parents=True, exist_ok=True)

                return local_file.write_text(file_data)
        except FileLockTimeout as exc:
            raise RuntimeError(
                f"Cannot update file path {file_path},"
                f" another instance of this application currently holds the lock."
            ) from exc
        finally:
            lock.release()
            self.delete_file(lock.lock_file)

    def delete_file(self, file_path: FilePath):
        local_file = self.local_path(file_path)
        self.logger.warning(f"Deleting local file {file_path}")
        return local_file.unlink(missing_ok=True)

    def local_path(self, file_path: FilePath) -> Path:
        """
        Return the local path for a given file path.

        Args:
            file_path (FilePath): The file path to convert to a local path.

        Returns:
            Path: The resolved local path.

        Raises:
            RuntimeError: If the file path is absolute and cannot be resolved.
            RuntimeError: If the CLI is not being run locally and there is no top level directory to use with the file path.
        """
        path = Path(file_path)
        if path.is_absolute():
            return path.resolve()

        if self.tld is None:
            caller = get_caller()
            raise RuntimeError(
                f"[{caller}] CLI is not being run locally and has no top level directory to use with {file_path}"
            )

        return Path(self.tld, file_path).resolve()

    def local_path_exists(self, file_path: FilePath) -> Path:
        """
        Check if the local path exists.

        Args:
            file_path (FilePath): The file path to check.

        Returns:
            Path: The resolved local path.

        Raises:
            RuntimeError: If the file path is empty.
            NotADirectoryError: If the directory does not exist locally.
        """
        caller = get_caller()

        if is_nothing(file_path):
            raise RuntimeError(f"File path being checked from {caller} is empty")

        local_file_path = self.local_path(file_path)

        if not local_file_path.exists():
            raise NotADirectoryError(
                f"Directory {local_file_path} from {caller} does not exist locally"
            )

        return local_file_path

    def get_repository_dir(self, dir_path: FilePath) -> Path:
        """
        Get the directory path for the repository.

        Args:
            dir_path (FilePath): The directory path to use for the repository.

        Returns:
            Path: The resolved directory path for the repository.

        """
        if self.tld:
            repo_dir_path = self.local_path(dir_path)
            repo_dir_path.mkdir(parents=True, exist_ok=True)
        else:
            repo_dir_path = Path(dir_path)

        return repo_dir_path

    def get_unique_sub_path(self, dir_path: FilePath):
        """
        Generate a unique sub path within the given directory path.

        Args:
            dir_path (FilePath): The directory path to generate the unique sub path within.

        Returns:
            Path: The resolved unique sub path.

        """
        local_dir_path = self.local_path(dir_path)

        def get_sub_path():
            return local_dir_path.joinpath(str(uuid.uuid1()))

        local_sub_path = get_sub_path()

        while local_sub_path.exists():
            local_sub_path = get_sub_path()

        return local_sub_path

    def get_rel_to_root(self, dir_path: FilePath) -> Optional[Path]:
        """
        Get the relative path from the top level directory to the given directory path.

        Args:
            dir_path (FilePath): The directory path to calculate the relative path to.

        Returns:
            Optional[Path]: The relative path from the top level directory to the given directory path, or None if the calculation fails.

        """
        try:
            return self.tld.relative_to(dir_path)
        except (ValueError, AttributeError):
            self.logger.warning(
                f"Could not calculate path for directory {dir_path} relative to the repository TLD {self.tld}",
                exc_info=True,
            )
            return None
