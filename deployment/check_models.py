import os, sys

MODEL_FILES = {
    "EEG_PD": r"E:\\Shiv Projects\\AgenticDiseaseFinder\\models\\Parkinson_Model.py",
    "MRI": r"E:\\Shiv Projects\\AgenticDiseaseFinder\\models\\adaptive_multi_scale_fusion_network.h5",
    "Neuroformer": r"E:\\Shiv Projects\\AgenticDiseaseFinder\\models\\neuroformer\\best_model.h5",
    "SPECTRA_SZ": r"E:\\Shiv Projects\\AgenticDiseaseFinder\\models\\SPECTRA-SZ\\spectra_run\\best_model.pt",
}

missing = [name for name, path in MODEL_FILES.items() if not os.path.isfile(path)]
if missing:
    print("Missing model files:", ", ".join(missing))
    sys.exit(1)
print("All required model files are present.")
