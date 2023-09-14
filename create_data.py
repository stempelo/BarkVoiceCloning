import os.path
import random
import uuid

import numpy

from bark import text_to_semantic
from bark.generation import load_model

from data import load_books, random_split_chunk
from IPython.lib.display import isdir

#Parameters
pathDataDrive = '/content/drive/MyDrive/ColabSpace/VoiceCloning/DataVoiceClone'
extensionFile = '.txt'
output = 'output'

print('Loading semantics model')
load_model(use_gpu=True, use_small=False, force_reload=False, model_type='text')



if os.path.isdir(pathDataDrive):
  print("Path exist on Drive")


if not os.path.isdir(output):
  os.mkdir(output)

contenuto_directory = os.listdir(pathDataDrive)
file_con_estensione_desiderata = [file for file in contenuto_directory if file.endswith(extensionFile)]
print(file_con_estensione_desiderata)

for file in file_con_estensione_desiderata:
    print(os.path.join(pathDataDrive, file))  # Stampa il percorso completo del file
    filetxt = os.path.join(pathDataDrive, file)
    filename = file[:-4] + '.npy'
    print(filename)
    file_name = os.path.join(output, filename)
    print(file_name)
    with open(filetxt, 'r') as file:
        # Read the entire file content
        content = file.read()
        text = content.strip()
        print("File content:")
        print(text)
    print('Generating semantics for text...')
    semantics = text_to_semantic(text, temp=round(random.uniform(0.6, 0.8), ndigits=2))
    numpy.save(file_name, semantics)

