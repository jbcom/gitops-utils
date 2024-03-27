import shlex
import subprocess


def get_process_output(cmd: str, timeout: int = None) -> (str, str):
    """
    Get the output of a process executed with the given command.

    Args:
        cmd (str): The command to execute.
        timeout (int): The maximum time in seconds to wait for the process to complete. If the process exceeds the timeout, it will be terminated.

    Returns:
        tuple: A tuple containing the stdout and stderr of the process.

    Raises:
        subprocess.CalledProcessError: If an error occurs while executing the command.
        subprocess.TimeoutExpired: If the process times out while executing the command.
    """
    result = subprocess.run(
        shlex.split(cmd),
        capture_output=True,
        text=True,
        shell=True,
        check=True,
        timeout=timeout,
    )
    return result.stdout, result.stderr
