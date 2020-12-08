import os
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import re


# Smiles to mol type
def smile_to_mol(smile, name="None"):
    try:

        m = Chem.MolFromSmiles(smile)
        m2 = Chem.AddHs(m)
        m2.SetProp("_Name", name)
        AllChem.EmbedMolecule(m2, randomSeed=0xf00d)
    except:
        m2 = None
    return m2


def get_coordinate(string, head=3):
    i = 0
    name_x_y_z = []
    for line in string.splitlines():
        if i > head and len(line) > 60:
            (x, y, z) = re.findall(r"\d+\.?\d*", line)[0:3]
            name = line[30:32]
            name_x_y_z.append([name, x, y, z])
        i = i + 1
    return name_x_y_z
