import importlib_metadata
from packaging import version
from db_graph.utils.logger import logger

def check_library_version(
    library_name: str,
    required_version: str,
    must_consistent: bool = False,
) -> bool:
  required_version = version.parse(required_version)
  try:
    installed_version = version.parse(importlib_metadata.version(library_name))
    if must_consistent:
      if installed_version != required_version:
        logger.error(f'{library_name} is installed and the version is {library_name}, \
                    but version {required_version} is required.')
        return False
    else:
      if installed_version < required_version:
        logger.error(f'{library_name} is installed and the version is {library_name}, \
                    but version {required_version} or higher is required.')
        return False
    return True
  except:
    logger.error(f'{library_name} is not installed, version {required_version} or higher is required.')
    return False