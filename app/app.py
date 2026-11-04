import streamlit as st
from rdkit import Chem

st.title("Toxicity from SMILES Prediction")

smiles = st.text_input("Input SMILES string:")
valid = bool(smiles) and Chem.MolFromSmiles(smiles) is not None

if smiles and not valid:
    st.error("The SMILES string is not valid.")

st.button("Submit", disabled=not valid)
