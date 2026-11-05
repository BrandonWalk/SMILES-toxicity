from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from skl2onnx import to_onnx
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

RDLogger.DisableLog("rdApp.*")   # silence RDKit parse warnings for invalid SMILES

NAME = "PEI-YU-HSUEH-ANISHA-JAISWAL-SRISH-MAULIK-BRANDON-WALKER"

THRESHOLD = 0.38

SEED = 2
FP_SIZE = 2048
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)

# Basic (protonatable) amines are the classic hERG pharmacophore; exclude amides, anilines,
# sulfonamides and nitriles, which are not basic.
BASIC_AMINE = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=[O,S,N]);!$(N-a);!$(N-[SX4]);!$(NC#N)]")
CATION = Chem.MolFromSmarts("[N+;!$(N=O);!$(N-[O-])]")
ACID = Chem.MolFromSmarts("[$([CX3](=O)[OX2H1,OX1-]),$([SX4](=O)(=O)[OX2H1,OX1-]),$(c1nn[nH]n1)]")
N_DESCRIPTORS = 17


def descriptors(mol):
    """Cheap physicochemical descriptors known to drive hERG binding (lipophilicity, size, basicity)."""
    return [Crippen.MolLogP(mol), Crippen.MolMR(mol), rdMolDescriptors.CalcTPSA(mol),
            Descriptors.MolWt(mol), mol.GetNumHeavyAtoms(), Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol), rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol), rdMolDescriptors.CalcNumHeteroatoms(mol),
            len(mol.GetSubstructMatches(BASIC_AMINE)), len(mol.GetSubstructMatches(CATION)),
            len(mol.GetSubstructMatches(ACID)),
            sum(atom.GetSymbol() in ("F", "Cl", "Br", "I") for atom in mol.GetAtoms()),
            Chem.GetFormalCharge(mol)]


def featurize(smiles):
    """SMILES strings -> (n, 2048 + 17) float32 array.

    Each molecule is reduced to its largest fragment (drops counter-ions such as HCl),
    then encoded as a 2048-bit Morgan count fingerprint (radius 2) plus 17 descriptors.
    Both have a fixed length, so SMILES of any length map to the same feature space.
    Unparseable SMILES get an all-zero row.
    """
    rows = np.zeros((len(smiles), FP_SIZE + N_DESCRIPTORS), np.float32)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        mol = max(Chem.GetMolFrags(mol, asMols=True), key=lambda frag: frag.GetNumHeavyAtoms())
        rows[i, :FP_SIZE] = MORGAN.GetCountFingerprintAsNumPy(mol)
        rows[i, FP_SIZE:] = descriptors(mol)
    return np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)


def build_model():
    """TODO 4: Return your model object here.

    If you are using sklearn, then an unfitted sklearn classifier or Pipeline,
    e.g. RandomForestClassifier(...).

    Not using sklearn? Export your own ONNX with one input of shape
    (n_molecules, n_features) where n_molecules can vary (PyTorch: dynamic_axes),
    and a last output holding P(blocker), with sigmoid/softmax inside the model.
    featurize must take SMILES strings and return one row per molecule.
    TestModel.py checks this.
    """
    return ExtraTreesClassifier(n_estimators=175, max_features=0.1, min_samples_leaf=2,
                                n_jobs=-1, random_state=SEED)


def save_onnx(model, sample):
    """
    Export a fitted model. `sample` is one row of your features,
    which fixes the input type.
    """
    Path("Part1_%s.onnx" % NAME).write_bytes(
        to_onnx(model, sample, options={"zipmap": False}, target_opset=12).SerializeToString())
    print("\nwrote Part1_%s.onnx" % NAME)


def load_split():
    """
    Load in assignment data and split into train/val/test sets
    You do not need to use this exact split, as you will be evaluated
    on a held-out test set. You may use the data in any way you wish.
    """
    df = pd.read_csv("data/hw1data.tab", sep="\t")
    df.columns = ["Drug_ID", "Drug", "Y"]
    test = df.sample(frac=0.2, random_state=35)
    rest = df.drop(test.index)
    valid = rest.sample(frac=0.125, random_state=1)
    return {"train": rest.drop(valid.index), "valid": valid, "test": test}


def prepare(frame):
    """Hand SMILES strings and labels to your featurizer."""
    return np.asarray(featurize(frame["Drug"].tolist()), np.float32), frame["Y"].to_numpy()


def report(y, probability, threshold):
    pred = (probability >= threshold).astype(int)
    tn, fp, _, _ = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print("\nthreshold %.2f" % threshold)
    for name, value in [("accuracy", accuracy_score(y, pred)),
                        ("sensitivity/recall", recall_score(y, pred)),
                        ("specificity", tn / (tn + fp)),
                        ("precision", precision_score(y, pred)),
                        ("f1", f1_score(y, pred)),
                        ("ROC-AUC", roc_auc_score(y, probability)),
                        ("PR-AUC", average_precision_score(y, probability))]:
        print("  %-20s%.3f" % (name, value))


def main():
    """
    You do not have to use the train/test split provided to you.
    This is just an example of splitting the data.
    You may use the entire hw1data.tab as you wish.
    The dataset used to evaluate your model is not provided to you (just
    like in the real world - you train on some data then deploy your model,
    which will receive new unseen data).
    """
    split = load_split()
    X_train, y_train = prepare(pd.concat([split["train"], split["valid"]]))
    X_test, y_test = prepare(split["test"])

    model = build_model().fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    report(y_test, probability, 0.5)
    report(y_test, probability, THRESHOLD)
    save_onnx(model, X_test[:1])


if __name__ == "__main__":
    main()