import os
import json
import shutil
from typing import Any

def make_dir(path: str):
  try:
    os.makedirs(path)
  except:
    pass

def remove_file(path: str):
  try:
    os.remove(path)
  except:
    print("WARNING: The file[{path}] to be deleted does not exist.".format(path=path))

def load_json(path: str):
  result = None
  try:
    fin = open(path)
    result = json.load(fin)
  except IOError as err:
    print("ERROR:", err)
  finally:
    if 'fin' in locals():
      fin.close()
  return result

def save_json(path: str, data: Any):
  try:
    fout = open(path, 'w')
    json.dump(data, fout, indent=4, sort_keys=True)
  except IOError as err:
    print("ERROR:", err)
  finally:
    if 'fout' in locals():
      fout.close()

def load_string_lines(path: str):
  result = []
  try:
    fin = open(path, 'r')
    result = fin.readlines()
  except IOError as err:
    print("ERROR:", err)
  finally:
    if 'fin' in locals():
      fin.close()
  return result

def load_string(path: str, return_input_on_error=False):
  result = ''
  try:
    fin = open(path, 'r')
    result = ''.join(fin.readlines()).strip().rstrip('\n')
  except IOError as err:
    if return_input_on_error:
      if 'fin' in locals():
        fin.close()
      return path
    print("ERROR:", err)
  finally:
    if 'fin' in locals():
      fin.close()
  return result

def save_string(path: str, data: str):
  try:
    fout = open(path, 'w')
    fout.write(data)
  except IOError as err:
    print("ERROR:", err)
  finally:
    if 'fout' in locals():
      fout.close()