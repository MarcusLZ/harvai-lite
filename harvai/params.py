import os


def get_path_data(path):
    if os.path.basename(path) == 'harvai-lite':
        return "raw_data/LEGITEXT000006074228.pdf"
    else:
        return "../raw_data/LEGITEXT000006074228.pdf"

def get_path_json(path):
    if os.path.basename(path) == 'harvai-lite':
        return "raw_data/data_preproc.json"
    else:
        return "../raw_data/data_preproc.json"

def get_path_json_digits(path):
    if os.path.basename(path) == 'harvai-lite':
        return "raw_data/data_preproc_digits.json"
    else:
        return "../raw_data/data_preproc_digits.json"

def get_path_faiss(path):
    if os.path.basename(path) == 'harvai-lite':
        return "raw_data/faiss.index"
    else:
        return "../raw_data/faiss.index"

def get_path_retriever(path):
    if os.path.basename(path) == 'harvai-lite':
        return "raw_data/retriever.pt"
    else:
        return "../raw_data/retriever.pt"
