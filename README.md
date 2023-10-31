# TopPeaks
In silico spectral library generation for data independent acquisition mass spectrum

# Installation & Run
1. Download and install original prosit source code from <a href="https://github.com/kusterlab/prosit">PROSIT</a><br>
2. Override our prosit source code to the original prosit source code<br>
3. Set MODEL_SPECTRA and MODEL_IRT paths in prosit/run_prosit_top_model.sh<br>
4. Run run_prosit_top_model.sh<br>

# Train top models
1. Download original prosit training datasets (you can get hint from <a href="https://github.com/kusterlab/prosit">PROSIT</a><br>
2. Once you copmlete your donwload, make a copy of the datasets because the next step will override top N training dataset in the original dataset<br>
3. Modify parameter "N" and set the original training dataset path in lib/HDF_data_generation.py<br>


