from periodictable import elements
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from os import cpu_count


# Smiles to mol type
def smile_to_mol(smile, name="None"):
    try:

        m = Chem.MolFromSmiles(smile)
        m2 = Chem.AddHs(m)
        m2.SetProp("_Name", name)
        AllChem.EmbedMolecule(m2, randomSeed=0xf00d)
        AllChem.UFFOptimizeMolecule(m2)
    except:
        m2 = None
    return m2


def smile_to_mol_df(df):
    return df.apply(smile_to_mol)


def mol_to_text(mol):
    if mol is None:
        return None
    try:
        i = Chem.MolToMolBlock(mol)
    except:
        i = None
    return i


def get_coordinate(string, head=3):
    try:
        i = 0
        name_x_y_z = []
        for line in string.splitlines():
            if i > head and len(line) > 60:
                (x, y, z) = [float(line[1:10]), float(line[11:20]), float(line[21:31])]
                name = line[30:32]
                charge = elements.symbol(name.strip()).number
                name_x_y_z.append([name, x, y, z, charge])
            i = i + 1
        coord = name_x_y_z
    except:
        coord = None
    return coord


def coulomb_matrix(coord, eig=True):
    try:
        c_mat = np.eye(len(coord))
        for i in range(len(coord)):
            c_mat[i, i] = 0.5 * coord[i][4] ** 2.4
            for j in range(i + 1, len(coord)):
                r = ((coord[i][1] - coord[j][1]) ** 2 + (coord[i][2] - coord[j][2]) ** 2 + (
                        coord[i][3] - coord[j][3]) ** 2
                     ) ** 0.5
                c_mat[i, j] = coord[i][4] * coord[j][4] / r
                c_mat[j, i] = c_mat[i, j]
        if eig == False:
            cm = c_mat
        else:
            cm = np.linalg.eig(c_mat)[0]
    except:
        cm = None
    return cm


def max_num(data):
    max = 0
    for i in data:
        if i.max() > max:
            max = i.max()
    return max


def vec_resize(vec, size):
    diff = size - len(vec)
    if diff % 2 == 0:
        return np.pad(vec, int(diff / 2))
    else:
        return np.pad(vec, (int((diff + 1) / 2), int((diff - 1) / 2)))


def apply_parallel(df, func, n=None):
    """
       @params df:
       @params func:      apply
       @params n:         n processor
       @return Dataframe:
　　 """

    if n is None:
        n = -1
    dflength = len(df)
    cpunum = cpu_count()
    if n < 0:
        spnum = cpunum + n + 1
    else:
        spnum = n or 1

    sp = list(range(dflength)[::int(dflength / spnum + 0.5)])
    sp.append(dflength)
    slice_gen = (slice(*idx) for idx in zip(sp[:-1], sp[1:]))
    results = Parallel(n_jobs=n)(delayed(func)(df[slc]) for slc in slice_gen)
    return pd.concat(results)

def get_mol_shape(mol,box=(10,10,10)):
    try:
        shape_obj=Chem.AllChem.ComputeMolShape(mol, boxDim=box)
        len_shape=(shape_obj.GetSize())
        shape=[]
        for i in range(0,len_shape):
            shape.append(shape_obj.GetVal(i))
    except:
        shape=None
    return shape


def atom_filter(mol, ban_list=False , atomic_range=False):

    ''' 
    Filter mol with ban element and atomic number range.
    Return True for mol pass the filter.
    
    Args:
        mol: rdkit mol obj
        ban_list: list of element symbol as ['Na','Ca','B']
        atomic_range: range of atomic_number  (0,2) would only allow H an He, (12,12) only allow  C
    # TODO opt the speed of this func
    '''
    if atomic_range != False:
         for i in mol.GetAtoms():
                if i.GetAtomicNum() > max(atomic_range) or i.GetAtomicNum() < min(atomic_range):
                    return False
   
                
    if ban_list !=False:
        for i in reversed(mol.GetAtoms()):                      
            if i.GetSymbol() in ban_list:
                return False

    return True


def filter_R(mol,patt):
    ''' 
    Filter mol with R group via SMARTS string 
    Return True for mol pass the filter.
    Args:
        mol: rdkit mol obj
        patt: SMARTS list of string for function group eg. {'N[H]':0 , '[H]C=O':1} for zero number of non-aromatic amine and only one aldehyde
    '''
    for i in patt:
        smarts=Chem.MolFromSmarts(i)
        if len(mol.GetSubstructMatches(smarts)) != patt[i]:
            return False
    return True