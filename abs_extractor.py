import os

import fitz
from tqdm import tqdm

from utils import show_time, write_to_json


def process_single_paper(path):
    doc = fitz.open(path)
    full_text = chr(12).join([page.get_text() for page in doc])
    abstract_start_idx = max(full_text.find('Abstract\n'), full_text.find('ABSTRACT\n'))
    intro_start_idx = max(full_text.find('Introduction\n', abstract_start_idx),
                          full_text.find('INTRODUCTION\n', abstract_start_idx))
    ref_start_idx = max(full_text.find('References\n', intro_start_idx),
                        full_text.find('REFERENCES\n', intro_start_idx))
    conclusion_start_idx = max(full_text.rfind('Conclusion', intro_start_idx, ref_start_idx),
                               full_text.rfind('CONCLUSION', intro_start_idx, ref_start_idx))

    abstract = full_text[
               abstract_start_idx: intro_start_idx] if abstract_start_idx != -1 and intro_start_idx != -1 else "Abstract not found"
    main_content_without_conclusion = full_text[
                   intro_start_idx: conclusion_start_idx if conclusion_start_idx != -1 else ref_start_idx] if intro_start_idx != -1 else "Main content not found"
    main_content = full_text[intro_start_idx: ref_start_idx] if intro_start_idx != -1 else "Main content not found"

    return abstract.strip().replace('\n', ' ').replace('- ', ''), main_content_without_conclusion.strip().replace('\n', ' ').replace('- ', ''), main_content.strip().replace('\n', ' ').replace('- ', '')


def process_papers(folder_path):
    papers_data = {}
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

    for filename in tqdm(pdf_files, desc="Processing PDFs"):
        path = os.path.join(folder_path, filename)
        print("{} Processing {}".format(show_time(), filename))
        abstract, main_text = process_single_paper(path)
        papers_data[filename.split(".")[0]] = {
            "abstract": abstract,
            "main_content": main_text
        }
        break

    return papers_data


if __name__ == '__main__':
    folder_path = './'  # Replace with your own path
    papers_data = process_papers(folder_path)
    write_to_json(papers_data, 'papers_data.json')




