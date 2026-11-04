"""Streamlit web app: predict hERG toxicity (channel blocking) from a SMILES string.

The featurization below must stay identical to `featurize` in the Part1 training
script, because the ONNX model was trained on exactly these 2048 + 17 features.
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import streamlit as st
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Draw, Lipinski, rdFingerprintGenerator, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

MODEL_PATH = Path(__file__).parent / "model.onnx"
THRESHOLD = 0.40          # same decision threshold as the training script

# ---- featurization (copied from the Part1 training script) -------------------------------
FP_SIZE = 2048
N_DESCRIPTORS = 17
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)
BASIC_AMINE = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=[O,S,N]);!$(N-a);!$(N-[SX4]);!$(NC#N)]")
CATION = Chem.MolFromSmarts("[N+;!$(N=O);!$(N-[O-])]")
ACID = Chem.MolFromSmarts("[$([CX3](=O)[OX2H1,OX1-]),$([SX4](=O)(=O)[OX2H1,OX1-]),$(c1nn[nH]n1)]")


def descriptors(mol):
    return [Crippen.MolLogP(mol), Crippen.MolMR(mol), rdMolDescriptors.CalcTPSA(mol),
            Descriptors.MolWt(mol), mol.GetNumHeavyAtoms(), Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol), rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol), rdMolDescriptors.CalcNumHeteroatoms(mol),
            len(mol.GetSubstructMatches(BASIC_AMINE)), len(mol.GetSubstructMatches(CATION)),
            len(mol.GetSubstructMatches(ACID)),
            sum(atom.GetSymbol() in ("F", "Cl", "Br", "I") for atom in mol.GetAtoms()),
            Chem.GetFormalCharge(mol)]


def largest_fragment(mol):
    """Drop counter-ions (e.g. the Cl in 'drug.Cl') and keep the biggest fragment."""
    return max(Chem.GetMolFrags(mol, asMols=True), key=lambda frag: frag.GetNumHeavyAtoms())


def featurize(smiles):
    rows = np.zeros((len(smiles), FP_SIZE + N_DESCRIPTORS), np.float32)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        mol = largest_fragment(mol)
        rows[i, :FP_SIZE] = MORGAN.GetCountFingerprintAsNumPy(mol)
        rows[i, FP_SIZE:] = descriptors(mol)
    return np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)


# ---- model ----------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_session():
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    dtype = {"tensor(double)": np.float64, "tensor(int64)": np.int64}.get(inp.type, np.float32)
    return session, inp.name, dtype


def predict_probability(smiles):
    """P(hERG blocker) for one SMILES string, read from the model's last output."""
    session, input_name, dtype = load_session()
    features = featurize([smiles]).astype(dtype)
    output = np.asarray(session.run(None, {input_name: features})[-1], float)
    probability = output[:, 1] if output.ndim == 2 and output.shape[1] == 2 else output.ravel()
    return float(np.clip(probability[0], 0.0, 1.0))


# ---- UI -------------------------------------------------------------------------------------
st.set_page_config(page_title="SMILES Toxicity Prediction", page_icon=":test_tube:")
st.title("Toxicity from SMILES Prediction")
st.caption("Predicts the probability that a molecule is a hERG channel blocker "
           "(a common cause of cardiotoxicity) using an ExtraTrees model exported to ONNX.")

smiles = st.text_input("Input SMILES string:", placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")
mol = Chem.MolFromSmiles(smiles) if smiles else None
valid = mol is not None and mol.GetNumAtoms() > 0

if smiles and not valid:
    st.error("The SMILES string is not valid.")

if st.button("Submit", type="primary", disabled=not valid) and valid:
    probability = predict_probability(smiles)
    toxic = probability >= THRESHOLD

    left, right = st.columns([1, 1])
    with left:
        st.image(Draw.MolToImage(largest_fragment(mol), size=(300, 300)), caption=smiles)
    with right:
        st.metric("P(hERG blocker)", f"{probability:.1%}")
        st.progress(probability)
        if toxic:
            st.error(f"**Predicted toxic** — probability {probability:.3f} is at or above the "
                     f"{THRESHOLD:.2f} decision threshold.")
        else:
            st.success(f"**Predicted non-toxic** — probability {probability:.3f} is below the "
                       f"{THRESHOLD:.2f} decision threshold.")

    if len(Chem.GetMolFrags(mol)) > 1:
        st.info("The input contains more than one fragment (e.g. a salt). "
                "Only the largest fragment is scored, as in training.")

    with st.expander("Molecular descriptors used by the model"):
        names = ["LogP", "Molar refractivity", "TPSA", "Molecular weight", "Heavy atoms",
                 "H-bond donors", "H-bond acceptors", "Rotatable bonds", "Rings", "Aromatic rings",
                 "Fraction sp3 C", "Heteroatoms", "Basic amines", "Cations", "Acids",
                 "Halogens", "Formal charge"]
        values = descriptors(largest_fragment(mol))
        st.table({"Descriptor": names,
                  "Value": [f"{v:.3f}" if isinstance(v, float) else str(v) for v in values]})
