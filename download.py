import os
import time
import pickle
import arxiv
from tqdm import tqdm
import argparse
import datetime
import json

from utils import show_time, write_to_pkl, write_to_json, get_llm_response_via_api
from abs_extractor import process_single_paper
from construct_relation_graph import author_reformat
from section_region_extractor import detect_region_and_annotate


def download(ids):
    # ROOT = "D:\\AcademicEval\\co_author"
    ROOT = "./"
    save_data = dict()
    # with open('./main_text_all_tmp.pkl', 'rb') as f:
    #     save_data_judge = pickle.load(f)
    # with open("{}".format(filename), "r", encoding='utf-8') as f:
    #     ids = f.readlines()

    client = arxiv.Client()
    for cnt, id in enumerate(tqdm(ids, desc="Processing ids", unit="id")):
        id = id.strip()
        # if id in save_data_judge:
        #     continue
        search = arxiv.Search(id_list=[id.split("/")[-1]], max_results=1)
        try:
            result = next(client.results(search), None)
        except:
            client = arxiv.Client()
            continue
        if result:
            print("Processing Paper: ", result.title)
            identifier = result.entry_id.split("/")[-1]
            TOL = 1
            while TOL:
                try:
                    result.download_pdf(dirpath=os.path.join(ROOT, "pdf"), filename=identifier + '.pdf')
                    break
                except:
                    time.sleep(5)
                    TOL -= 1
            if TOL == 0:
                print("Skip")
                continue
            try:
                # _, main_text = process_single_paper(os.path.join(ROOT, "pdf", identifier + '.pdf'))
                _, main_text_without_conclusion, main_text = process_single_paper(os.path.join(ROOT, "pdf", identifier + '.pdf'))
                if main_text == "Main content not found":
                    main_text = ""
                if main_text_without_conclusion == "Main content not found":
                    main_text_without_conclusion = ""
                regions, intro, main_text_without_intro = detect_region_and_annotate(os.path.join(ROOT, "pdf", identifier + '.pdf'), section="introduction", save=False)
                if len(regions) == 0:
                    intro = ""
                    main_text_without_intro = ""
                regions, related, main_text_without_related = detect_region_and_annotate(os.path.join(ROOT, "pdf", identifier + '.pdf'), section="related work", save=False)
                if len(regions) == 0:
                    related = ""
                    main_text_without_related = ""
            except:
                try:
                    os.remove(os.path.join(ROOT, "pdf", identifier + '.pdf'))
                    continue
                except:
                    continue
            save_data[id] = {"main_content": main_text, "main_content_without_conclusion": main_text_without_conclusion,
                             "main_content_without_intro": main_text_without_intro,
                             "main_content_without_related": main_text_without_related,
                             "introduction": intro, "related": related}
            try:
                os.remove(os.path.join(ROOT, "pdf", identifier + '.pdf'))
            except:
                pass

        if cnt % 100 == 0:
            write_to_pkl(save_data, "./AcademicEval/co_author/main_text_all.pkl")

    write_to_pkl(save_data, "./AcademicEval/co_author/main_text_all.pkl")


if __name__ == '__main__':
    pass

