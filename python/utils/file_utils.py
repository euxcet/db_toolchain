import os
import json
import shutil

def make_dir(path):
  try:
    os.makedirs(path)
  except:
    pass

def remove_file(path):
  try:
    os.remove(path)
  except:
    print("WARNING: The file[{path}] to be deleted does not exist.".format(path=path))

def load_json(path):
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

def save_json(path, data):
  try:
    fout = open(path, 'w')
    json.dump(data, fout)
  except IOError as err:
    print("ERROR:", err)
  finally:
    if 'fout' in locals():
      fout.close()

def load_string_lines(path):
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

def load_string(path, return_input_on_error=False):
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

def save_string(path, data):
  try:
    fout = open(path, 'w')
    fout.write(data)
  except IOError as err:
    print("ERROR:", err)
  finally:
    if 'fout' in locals():
      fout.close()