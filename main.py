import os
import shutil
import json
import pickle
import time
import random
import datetime
from multiprocessing import Pool
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np

import arxiv

from utils import *
from abs_extractor import process_single_paper
from section_region_extractor import detect_region_and_annotate
from related_extractor import extract_related_and_ref, split_refs, extract_refs_info, get_related_titles, \
    fetch_related_papers_arxiv, finish_dataset, search_arxiv_by_title
from construct_relation_graph import bfs, author_reformat
from refine_graph import check_connectivity, get_max_connect_part, stats


def get_random_papers_via_arXiv(main_field, save_path, num_papers_per_subfield=100):
    # subfields = ["fairness", "reasoning", "weather predicting", "text generation", "diffusion model",
    #              "machine translation", "sentiment analysis", "speech recognition", "multimodal", "dialogue systems",
    #              "language modeling", "fake news detection", "social NLP", "text summariztation", "question answering",
    #              "emotion recognition in text", "lexical semantics", "information extraction", "psychology",
    #              "drug design"]
    client = arxiv.Client()
    all_papers_info = []

    # for subfield in subfields:
    print("{} Fetching Info for Subfield: {}".format(show_time(), main_field))
    # search_query = "cat:{} AND {}".format(main_field, subfield)
    search_query = "{}".format(main_field)
    print("Search Query: ", search_query)
    search = arxiv.Search(
        query=search_query,
        max_results=num_papers_per_subfield
    )
    try:
        results = client.results(search)
    except:
        client = arxiv.Client()
        results = client.results(search)
    results_list = list(results)
    print("Number of Results Found: {}".format(len(results_list)))

    for result in results_list:
        print("Processing Paper: ", result.title)
        paper_info = {
            'url': result.entry_id,
            'title': result.title,
            'abstract': result.summary,
            'authors': ', '.join(author.name for author in result.authors),
            'published': str(result.published).split(" ")[0],
            'updated': str(result.updated).split(" ")[0],
            'primary_cat': result.primary_category,
            'cats': result.categories,
            'category': main_field
        }
        all_papers_info.append(paper_info)

    check_path(save_path)
    if num_papers_per_subfield != 100:
        write_to_json(all_papers_info, os.path.join(save_path, "random_paper_abs_" + str(num_papers_per_subfield) + ".json"))
    else:
        write_to_json(all_papers_info, os.path.join(save_path, "random_paper_abs.json"))


def download_papers_via_arXiv(target_fields, save_path, num_papers_per_subfield=5):
    check_path(save_path)
    client = arxiv.Client()

    for subfield in target_fields:
        sub_save_path = os.path.join(save_path, subfield)
        pdf_sub_save_path = os.path.join(sub_save_path, "pdf")
        json_sub_save_path = os.path.join(sub_save_path, "json")
        check_path(sub_save_path)
        check_path(pdf_sub_save_path)
        check_path(json_sub_save_path)
        print("{} Downloading Raw Papers for Target Field: {}".format(show_time(), subfield))
        # search_query = "cat:{}".format(subfield)
        search_query = "{}".format(subfield)
        print("Search Query: ", search_query)
        search = arxiv.Search(
            query=search_query,
            max_results=num_papers_per_subfield,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        try:
            results = client.results(search)
        except:
            client = arxiv.Client()
            continue
        results_list = list(results)
        print("Number of Results Found: {}".format(len(results_list)))
        # TODO: pre-save

        for result in results_list:
            print("Processing Paper: ", result.title)
            paper_info = {
                'url': result.entry_id,
                'title': result.title,
                'abstract': result.summary,
                'authors': ', '.join(author.name for author in result.authors),
                'published': str(result.published).split(" ")[0],
                'updated': str(result.updated).split(" ")[0],
                'primary_cat': result.primary_category,
                'cats': result.categories,
                'label': "Original Paper"
            }
            identifier = result.entry_id.split("/")[-1]
            if os.path.exists(os.path.join(pdf_sub_save_path, identifier + '.pdf')) and \
                    os.path.exists(os.path.join(json_sub_save_path, identifier + '.json')):
                continue
            TOL = 1
            while TOL:
                try:
                    result.download_pdf(dirpath=pdf_sub_save_path, filename=identifier + '.pdf')
                    break
                except:
                    time.sleep(5)
                    TOL -= 1
            if TOL == 0:
                print("Skip")
                continue
            write_to_json(paper_info, os.path.join(json_sub_save_path, identifier + '.json'))

        if not os.path.exists(os.path.join(sub_save_path, "random_paper_abs.json")):
            get_random_papers_via_arXiv(main_field=subfield, save_path=sub_save_path)


def file_check(root):
    for paper_cat in os.listdir(os.path.join(root, "raw_papers")):
        pdf_files = [i[:-4] for i in os.listdir(os.path.join(root, "raw_papers", paper_cat, "pdf"))]
        json_files = [i[:-5] for i in os.listdir(os.path.join(root, "raw_papers", paper_cat, "json"))]
        remove_files = list(set(pdf_files).symmetric_difference(set(json_files)))
        print(remove_files)
        for rf in remove_files:
            rf_pdf_path = os.path.join(root, "raw_papers", paper_cat, "pdf", rf + ".pdf")
            rf_json_path = os.path.join(root, "raw_papers", paper_cat, "json", rf + ".json")
            try:
                os.remove(rf_pdf_path)
            except:
                pass
            try:
                os.remove(rf_json_path)
            except:
                pass


def file_date_check_deprecated(root, min_year=2024, min_month=1):
    for paper_cat in os.listdir(os.path.join(root, "raw_papers")):
        remove_files = []
        json_files = os.listdir(os.path.join(root, "raw_papers", paper_cat, "json"))
        for json_file_path in json_files:
            with open(os.path.join(root, "raw_papers", paper_cat, "json", json_file_path), 'r', encoding='utf-8') as file:
                json_data = json.load(file)
            date = json_data["published"].split("-")
            year = int(date[0])
            month = int(date[1])
            if year < min_year or month < min_month:
                remove_files.append(json_file_path[:-5])

        check_path(os.path.join(root, "raw_papers", paper_cat, "pdf_old"))
        check_path(os.path.join(root, "raw_papers", paper_cat, "json_old"))
        print(remove_files)
        for rf in remove_files:
            rf_pdf_path = os.path.join(root, "raw_papers", paper_cat, "pdf", rf + ".pdf")
            rf_json_path = os.path.join(root, "raw_papers", paper_cat, "json", rf + ".json")
            try:
                shutil.move(rf_pdf_path, os.path.join(root, "raw_papers", paper_cat, "pdf_old"))
            except:
                pass
            try:
                shutil.move(rf_json_path, os.path.join(root, "raw_papers", paper_cat, "json_old"))
            except:
                pass


def setting_file_date_check_deprecated(root, setting, min_year=2024, min_month=1):
    remove_files = []
    for json_file in os.listdir(os.path.join(root, setting)):
        with open(os.path.join(root, setting, json_file), 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        if setting == "related_multi":
            date = json_data[0]["published"].split("-")
        else:
            raise NotImplementedError("Error")
        year = int(date[0])
        month = int(date[1])
        if year < min_year or month < min_month:
            remove_files.append(json_file[:-5])

    check_path(os.path.join(root, setting + "_old"))
    print(remove_files)
    for rf in remove_files:
        rf_pdf_path = os.path.join(root, setting, rf + ".json")
        try:
            shutil.move(rf_pdf_path, os.path.join(root, setting + "_old"))
        except:
            pass


def filter_by_citation_num(root, min=15, max=20):
    remove_files = []
    for json_file in os.listdir(os.path.join(root, "related_multi")):
        if not json_file.endswith(".json"):
            continue
        with open(os.path.join(root, "related_multi", json_file), 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        cnt = 0
        for i in json_data:
            if "label" in i and i["label"] == "Related Work":
                cnt += 1

        if not (cnt >= min and cnt <= max):
            remove_files.append(json_file[:-5])

    check_path(os.path.join(root, "related_multi" + "_less"))
    print(remove_files, len(remove_files))
    for rf in remove_files:
        rf_pdf_path = os.path.join(root, "related_multi", rf + ".json")
        try:
            shutil.move(rf_pdf_path, os.path.join(root, "related_multi" + "_less"))
        except:
            pass


def compatible_transfer_deprecated(filename, root):
    sub_save_path = os.path.join(root, "raw_papers", "cs.CL")
    pdf_sub_save_path = os.path.join(sub_save_path, "pdf")
    json_sub_save_path = os.path.join(sub_save_path, "json")
    check_path(sub_save_path)
    check_path(pdf_sub_save_path)
    check_path(json_sub_save_path)
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)

    client = arxiv.Client()
    for ind, (k, v) in enumerate(data.items()):
        result = search_arxiv_by_title(client, k, tol=10)
        if result:
            print("{} Transfer Successful".format(show_time()))
            paper_info = {
                'url': result.entry_id,
                "title": result.title,
                "abstract": result.summary,
                "authors": ', '.join(author.name for author in result.authors),
                "published": str(result.published).split(" ")[0],
                "updated": str(result.updated).split(" ")[0],
                'primary_cat': result.primary_category,
                'cats': result.categories,
                "label": "Original Paper"
            }
            identifier = result.entry_id.split("/")[-1]
            if os.path.exists(os.path.join(pdf_sub_save_path, identifier + '.pdf')) and \
                    os.path.exists(os.path.join(json_sub_save_path, identifier + '.json')):
                continue
            result.download_pdf(dirpath=pdf_sub_save_path, filename=identifier + '.pdf')
            write_to_json(paper_info, os.path.join(json_sub_save_path, identifier + '.json'))
        else:
            print("{} Transfer Failed: {}".format(show_time(), k))


def construct_short_task(data):
    pdf_file, pdf_file_path, json_file_path, paper_cat, save_path, setting = data
    print("{} Construct {}-Short: {}".format(show_time(), setting.title(), pdf_file))
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    json_data["paper_cat"] = paper_cat
    try:
        if setting == "title":
            _, _, main_text = process_single_paper(pdf_file_path)
            if main_text == "Main content not found":
                return
            json_data["gt"] = json_data['title']
            json_data["main_content"] = main_text
        elif setting == "abstract":
            _, main_text, _ = process_single_paper(pdf_file_path)
            if main_text == "Main content not found":
                return
            json_data["gt"] = json_data['abstract']
            json_data["main_content"] = main_text
        elif setting == "introduction":
            regions, intro, main_text = detect_region_and_annotate(pdf_file_path, section="introduction", save=False)
            if len(regions) == 0 or len(intro) == 0 or len(main_text) == 0:
                return
            json_data["gt"] = intro
            json_data["main_content"] = main_text
        else:
            raise Exception("Error")
    except:
        return

    write_to_json(json_data, os.path.join(save_path, "{}_short_".format(setting) + pdf_file[:-4] + ".json"))


def construct_short(root, setting):
    save_path = os.path.join(root, "{}_short".format(setting))
    check_path(save_path)

    paper_cats = []
    pdf_file_list = []
    pdf_file_path_list = []
    json_file_path_list = []
    for paper_cat in os.listdir(os.path.join(root, "raw_papers")):
        for pdf_file in os.listdir(os.path.join(root, "raw_papers", paper_cat, "pdf")):
            pdf_file_path = os.path.join(root, "raw_papers", paper_cat, "pdf", pdf_file)
            json_file_path = os.path.join(root, "raw_papers", paper_cat, "json", pdf_file[:-4] + ".json")
            paper_cats.append(paper_cat)
            pdf_file_list.append(pdf_file)
            pdf_file_path_list.append(pdf_file_path)
            json_file_path_list.append(json_file_path)

    with Pool(40) as p:
        p.map(construct_short_task,
              [(x, y, z, e, save_path, setting) for x, y, z, e in zip(pdf_file_list, pdf_file_path_list, json_file_path_list, paper_cats)])


def construct_related_short_task(data):
    json_file, json_file_path, root, save_path, setting = data
    print("{} Construct {}-Short: {}".format(show_time(), setting.title(), json_file_path))
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    source_pdf_file_path = os.path.join(root, "raw_papers", json_data[0]["paper_cat"], "pdf", json_file.split("_")[-1].replace("json", "pdf"))
    try:
        regions, _, main_text = detect_region_and_annotate(source_pdf_file_path, section="related work", save=False)
        if len(regions) == 0 or len(main_text) == 0:
            return
        json_data[0]["main_content"] = main_text
    except:
        return

    write_to_json(json_data, os.path.join(save_path, "{}_short_".format(setting) + json_file.split("_")[-1]))


def construct_related_short(root):
    save_path = os.path.join(root, "{}_short".format("related"))
    check_path(save_path)

    json_file_list = []
    json_file_path_list = []
    for related_json_path in os.listdir(os.path.join(root, "related_multi")):
        if not related_json_path.endswith(".json"):
            continue
        json_file_list.append(related_json_path)
        json_file_path_list.append(os.path.join(root, "related_multi", related_json_path))

    with Pool(40) as p:
        p.map(construct_related_short_task, [(x, y, root, save_path, "related") for x, y in zip(json_file_list, json_file_path_list)])


def construct_long_task2(data):
    file_path, cat_file_path, save_path, setting, graph_info, both = data
    print("{} Construct {}-Long: {}".format(show_time(), setting, file_path))
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    cat_file_path_all = []
    for _, v in cat_file_path.items():
        cat_file_path_all.extend(v)

    if not isinstance(json_data, list):
        first_author = author_reformat(json_data["authors"].split(",")[0].strip())
        if json_data["paper_cat"] in cat_file_path:
            cat_files = cat_file_path[json_data["paper_cat"]]
        else:
            cat_files = cat_file_path_all
    else:
        first_author = author_reformat(json_data[0]["authors"].split(",")[0].strip())
        if json_data[0]["paper_cat"] in cat_file_path:
            cat_files = cat_file_path[json_data[0]["paper_cat"]]
        else:
            cat_files = cat_file_path_all
    if len(cat_files) <= 5:
        cat_files = cat_file_path_all
    if file_path in cat_files:
        cat_files.remove(file_path)
    selected_cat_files = np.random.choice(cat_files, size=5, replace=False)
    additional_info = []
    for selected_cat_file in selected_cat_files:
        with open(selected_cat_file, 'r', encoding='utf-8') as file:
            info = json.load(file)
        additional_info.append(info)

    if graph_info is not None:
        if len(graph_info) != 0:
            additional_graph_info = graph_info
            remove_authors = set()
            for k, v in additional_graph_info["node_feat"].items():
                new_v = []
                for sub_v in v:
                    pub_time = sub_v["published"].split("-")
                    pub_date = datetime.date(year=int(pub_time[0]), month=int(pub_time[1]), day=int(pub_time[2]))
                    val_cutoff = datetime.date(year=2024, month=3, day=25)
                    # sub_v["url"].find(file_path.split("_")[-1].replace(".json", "")) != -1
                    if pub_date > val_cutoff and sub_v["url"].find(file_path.split("_")[-1].replace(".json", "")) == -1:
                        continue
                    if setting == "title":
                        if len(sub_v["main_content"]) == 0:
                            continue
                        del sub_v["main_content_without_conclusion"]
                        del sub_v["main_content_without_intro"]
                        del sub_v["main_content_without_related"]
                        del sub_v["introduction"]
                        del sub_v["related"]
                    elif setting == "abstract":
                        if len(sub_v["main_content_without_conclusion"]) == 0:
                            continue
                        sub_v["main_content"] = sub_v["main_content_without_conclusion"]
                        del sub_v["main_content_without_conclusion"]
                        del sub_v["main_content_without_intro"]
                        del sub_v["main_content_without_related"]
                        del sub_v["introduction"]
                        del sub_v["related"]
                    elif setting == "introduction":
                        if len(sub_v["main_content_without_intro"]) == 0 or len(sub_v["introduction"]) == 0:
                            continue
                        sub_v["main_content"] = sub_v["main_content_without_intro"]
                        del sub_v["main_content_without_conclusion"]
                        del sub_v["main_content_without_intro"]
                        del sub_v["main_content_without_related"]
                        del sub_v["related"]
                    elif setting == "related":
                        if len(sub_v["main_content_without_related"]) == 0 or len(sub_v["related"]) == 0:
                            continue
                        sub_v["main_content"] = sub_v["main_content_without_related"]
                        del sub_v["main_content_without_conclusion"]
                        del sub_v["main_content_without_intro"]
                        del sub_v["main_content_without_related"]
                        del sub_v["introduction"]
                    else:
                        raise Exception("Error")
                    new_v.append(sub_v)
                if len(new_v) == 0:
                    remove_authors.add(k)
                additional_graph_info["node_feat"][k] = new_v

            remove_eid = []
            for eid, edge in enumerate(additional_graph_info["graph"]):
                a1 = edge[0]
                a2 = edge[1]
                if a1 in remove_authors or a2 in remove_authors:
                    remove_eid.append(eid)

            [additional_graph_info["graph"].pop(index) for index in sorted(remove_eid, reverse=True)]
            [additional_graph_info["node_feat"].pop(index) for index in remove_authors]

            if len(additional_graph_info["node_feat"]) == 0:
                additional_graph_info = {}
            else:
                if not check_connectivity(additional_graph_info):
                    additional_graph_info = get_max_connect_part(additional_graph_info, node_idx=0)
                paper_total = stats(additional_graph_info)
                if paper_total < 2:
                    additional_graph_info = {}
        else:
            additional_graph_info = {}


    if not isinstance(json_data, list):
        if graph_info is None:
            json_data["additional_info"] = additional_info
        elif not both:
            if len(additional_graph_info) == 0:
                return
            json_data["additional_graph_info"] = additional_graph_info
        else:
            json_data["additional_info"] = additional_info
            json_data["additional_graph_info"] = additional_graph_info
    else:
        if graph_info is None:
            json_data[0]["additional_info"] = additional_info
        elif not both:
            if len(additional_graph_info) == 0:
                return
            json_data[0]["additional_graph_info"] = additional_graph_info
        else:
            json_data[0]["additional_info"] = additional_info
            json_data[0]["additional_graph_info"] = additional_graph_info


    write_to_json(json_data, os.path.join(save_path, file_path.split("/")[-1].replace("short", "long")))


def construct_long_task1(data):
    with open(data, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    if isinstance(json_data, list):
        json_data = json_data[0]
    paper_cat = json_data["paper_cat"]
    pub = json_data["published"].split("-")
    pub_date = datetime.date(year=int(pub[0]), month=int(pub[1]), day=int(pub[2]))

    return data, paper_cat, json_data["published"], pub_date


def construct_long(root, setting, augmented_with_graph=False, both=False):
    if augmented_with_graph:
        if not both:
            save_path = os.path.join(root, "{}_long_graph".format(setting))
        else:
            save_path = os.path.join(root, "{}_long_mix".format(setting))
        with open("./subgraph_refine.pkl", 'rb') as f:
            graph_info = pickle.load(f)
    else:
        save_path = os.path.join(root, "{}_long".format(setting))
    check_path(save_path)

    train_info_path = os.path.join(root, "{}_short".format(setting), "train.txt")
    with open(train_info_path, "r", encoding="utf-8") as f:
        train_files = f.readlines()
    train_file_path = [os.path.join(root, "{}_short".format(setting), "{}_short_{}".format(setting, i.strip())) for i in train_files]

    val_info_path = os.path.join(root, "{}_short".format(setting), "val.txt")
    with open(val_info_path, "r", encoding="utf-8") as f:
        val_files = f.readlines()
    val_file_path = [os.path.join(root, "{}_short".format(setting), "{}_short_{}".format(setting, i.strip())) for i in val_files]

    test_info_path = os.path.join(root, "{}_short".format(setting), "test.txt")
    with open(test_info_path, "r", encoding="utf-8") as f:
        test_files = f.readlines()
    test_file_path = [os.path.join(root, "{}_short".format(setting), "{}_short_{}".format(setting, i.strip())) for i in test_files]


    train_file_cat_path = dict()
    with Pool(40) as p:
        ret = p.map(construct_long_task1, train_file_path)
    for train_file, paper_cat, pub_str, pub_date in ret:
        if paper_cat in train_file_cat_path:
            train_file_cat_path[paper_cat].append(train_file)
        else:
            train_file_cat_path[paper_cat] = []

    train_val_file_cat_path = dict()
    with Pool(40) as p:
        ret = p.map(construct_long_task1, train_file_path + val_file_path)
    for train_val_file, paper_cat, pub_str, pub_date in ret:
        if paper_cat in train_val_file_cat_path:
            train_val_file_cat_path[paper_cat].append(train_val_file)
        else:
            train_val_file_cat_path[paper_cat] = []


    with Pool(40) as p:
        p.map(construct_long_task2, [(x, train_file_cat_path, save_path, setting, graph_info[x.split("_")[-1].replace(".json", "")] if augmented_with_graph else None, both) for x in train_file_path + val_file_path])

    with Pool(40) as p:
        p.map(construct_long_task2, [(x, train_val_file_cat_path, save_path, setting, graph_info[x.split("_")[-1].replace(".json", "")] if augmented_with_graph else None, both) for x in test_file_path])


def construct_title_single_task(data):
    pdf_file, pdf_file_path, json_file_path, save_path = data
    print("{} Construct Title-Single: {}".format(show_time(), pdf_file))
    try:
        _, main_text = process_single_paper(pdf_file_path)
    except:
        return
    if main_text == "Main content not found":
        return
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    json_data["gt"] = json_data['title']
    json_data["pre_questions"] = []
    json_data["main_content"] = main_text
    write_to_json(json_data, os.path.join(save_path, "title_single_" + pdf_file[:-4] + ".json"))


def construct_title_single(root):
    save_path = os.path.join(root, "title_single")
    check_path(save_path)

    pdf_file_list = []
    pdf_file_path_list = []
    json_file_path_list = []
    for paper_cat in os.listdir(os.path.join(root, "raw_papers")):
        for pdf_file in os.listdir(os.path.join(root, "raw_papers", paper_cat, "pdf")):
            pdf_file_path = os.path.join(root, "raw_papers", paper_cat, "pdf", pdf_file)
            json_file_path = os.path.join(root, "raw_papers", paper_cat, "json", pdf_file[:-4] + ".json")
            pdf_file_list.append(pdf_file)
            pdf_file_path_list.append(pdf_file_path)
            json_file_path_list.append(json_file_path)

    with Pool(40) as p:
        p.map(construct_title_single_task,
              [(x, y, z, save_path) for x, y, z in zip(pdf_file_list, pdf_file_path_list, json_file_path_list)])


def construct_abstract_single_task(data):
    pdf_file, pdf_file_path, json_file_path, save_path = data
    print("{} Construct Abstract-Single: {}".format(show_time(), pdf_file))
    try:
        _, main_text = process_single_paper(pdf_file_path)
    except:
        return
    if main_text == "Main content not found":
        return
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    json_data["gt"] = json_data['abstract']
    json_data["pre_questions"] = []
    json_data["main_content"] = main_text
    write_to_json(json_data, os.path.join(save_path, "abstract_single_" + pdf_file[:-4] + ".json"))


def construct_abstract_single(root):
    save_path = os.path.join(root, "abstract_single")
    check_path(save_path)

    pdf_file_list = []
    pdf_file_path_list = []
    json_file_path_list = []
    for paper_cat in os.listdir(os.path.join(root, "raw_papers")):
        for pdf_file in os.listdir(os.path.join(root, "raw_papers", paper_cat, "pdf")):
            pdf_file_path = os.path.join(root, "raw_papers", paper_cat, "pdf", pdf_file)
            json_file_path = os.path.join(root, "raw_papers", paper_cat, "json", pdf_file[:-4] + ".json")
            pdf_file_list.append(pdf_file)
            pdf_file_path_list.append(pdf_file_path)
            json_file_path_list.append(json_file_path)

    with Pool(40) as p:
        p.map(construct_abstract_single_task, [(x, y, z, save_path) for x, y, z in zip(pdf_file_list, pdf_file_path_list, json_file_path_list)])


def split_batch(instructions, batch_size=5, drop_last=True):
    batch_instructions = []
    sub_batch = []
    for ind, ins in enumerate(instructions):
        if ind != 0 and ind % batch_size == 0:
            batch_instructions.append(sub_batch)
            sub_batch = [ins]
        else:
            sub_batch.append(ins)

    if len(sub_batch) == batch_size or (len(sub_batch) != 0 and not drop_last):
        batch_instructions.append(sub_batch)

    return batch_instructions


def construct_abstract_multi_task2(data):
    ind, batch_files, save_path = data
    print("{} Construct Abstract-Multi: ".format(show_time()), batch_files)
    save_data = []
    first_entry = {"id": str(ind), "published": batch_files[-1][1], "gt": [], "pre_questions": []}
    save_data.append(first_entry)
    for batch_file in batch_files:
        try:
            _, main_text = process_single_paper(batch_file[0])
        except:
            return
        with open(batch_file[0].replace("pdf", "json"), 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        json_data["main_content"] = main_text
        save_data.append(json_data)

    write_to_json(save_data, os.path.join(save_path, "abstract_multi_" + str(ind) + ".json"))


def construct_abstract_multi_task1(data):
    try:
        _, main_text = process_single_paper(data)
        if main_text == "Main content not found":
            raise Exception("Error")
        with open(data.replace("pdf", "json"), 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        pub = json_data["published"].split("-")
        pub_date = datetime.date(year=int(pub[0]), month=int(pub[1]), day=int(pub[2]))
        return data, json_data["published"], pub_date, 1
    except:
        return data, None, None, 0


def construct_abstract_multi(root):
    save_path = os.path.join(root, "abstract_multi")
    check_path(save_path)

    all_pdf_file_path = dict()
    for paper_cat in os.listdir(os.path.join(root, "raw_papers")):
        container = []
        for pdf_file in os.listdir(os.path.join(root, "raw_papers", paper_cat, "pdf")):
            pdf_file_path = os.path.join(root, "raw_papers", paper_cat, "pdf", pdf_file)
            container.append(pdf_file_path)

        with Pool(40) as p:
            ret = p.map(construct_abstract_multi_task1, container)
        container = [(i[0], i[1], i[2]) for i in ret if i[3] == 1]
        container.sort(key=lambda x: x[2], reverse=True)
        all_pdf_file_path[paper_cat] = container

    batch_pdf_file_path = []
    for _, v in all_pdf_file_path.items():
        batch_pdf_file_path.extend(split_batch(v, batch_size=5, drop_last=True))

    with Pool(40) as p:
        p.map(construct_abstract_multi_task2, [(ind, batch_files, save_path) for ind, batch_files in enumerate(batch_pdf_file_path)])


def construct_related_multi(root):
    target_fields = ["Spatio AND Temporal AND Data AND Mining", "LLM AND Jailbreak", "Mamba",
                     "Multi AND Modal AND LLM", "Diffusion AND Model",
                     "Mixture AND of AND Experts", "Parameter AND Efficient AND Fine AND Tuning",
                     "Retrieval AND Augmented AND Generation AND RAG", "Semantic AND Segmentation AND Image"]
    save_path = os.path.join(root, "related_multi")
    check_path(save_path)

    for paper_cat in os.listdir(os.path.join(root, "raw_papers")):
        if paper_cat not in target_fields:
            continue
        for pdf_file in os.listdir(os.path.join(root, "raw_papers", paper_cat, "pdf")):
            print("{} Construct Related-Multi: {} {}".format(show_time(), paper_cat, pdf_file))
            sample_save_path = os.path.join(save_path, "related_multi_" + pdf_file[:-4] + ".json")
            if os.path.exists(sample_save_path):
                continue
            pdf_file_path = os.path.join(root, "raw_papers", paper_cat, "pdf", pdf_file)
            json_file_path = os.path.join(root, "raw_papers", paper_cat, "json", pdf_file[:-4] + ".json")
            random_paper_file_path = os.path.join(root, "raw_papers", paper_cat, "random_paper_abs.json")
            with open(json_file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)

            try:
                _, main_text = process_single_paper(pdf_file_path)
                citations, refs, related_work_section = extract_related_and_ref(pdf_file_path)
                refs = split_refs(refs)
                refs = extract_refs_info(refs)
                related_titles = get_related_titles(citations, refs)
            except:
                continue

            if len(related_titles) < 5:
                continue

            json_data["paper_cat"] = paper_cat
            json_data["gt"] = related_work_section
            json_data["pre_questions"] = []
            json_data["main_content"] = main_text
            write_to_json([json_data], sample_save_path)

            try:
                fail_cases = fetch_related_papers_arxiv(related_titles, sample_save_path)
            except:
                os.remove(sample_save_path)
                continue
            if len(related_titles) - len(fail_cases) < 5:
                os.remove(sample_save_path)
                continue
            finish_dataset(sample_save_path, random_paper_file_path)


def construct_related_multi_papers(root, num=10):
    save_path = os.path.join(root, "related_multi_papers")
    check_path(save_path)

    gather_dict = dict()
    for single_paper in os.listdir(os.path.join(root, "related_multi")):
        if not single_paper.endswith(".json"):
            continue
        json_file_path = os.path.join(root, "related_multi", single_paper)
        with open(json_file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        # primary_cat = json_data[0]["primary_cat"]
        primary_cat = json_data[0]["paper_cat"]
        if primary_cat in gather_dict:
            gather_dict[primary_cat].append(json_data)
        else:
            gather_dict[primary_cat] = [json_data]

    cnt = 0
    for pri_cat, gather_data in gather_dict.items():
        get_random_papers_via_arXiv(main_field=pri_cat, save_path=os.path.join(root, "raw_papers", pri_cat), num_papers_per_subfield=1000)
        with open(os.path.join(root, "raw_papers", pri_cat, "random_paper_abs_1000.json"), 'r', encoding='utf-8') as file2:
            random_papers = json.load(file2)
        split_gather_data = split_batch(gather_data, batch_size=num, drop_last=True)
        for batch_gather_data in split_gather_data:
            sample_save_path = os.path.join(save_path, str(cnt) + '.json')
            save_json_data = dict()
            paper_info = []
            chunk_info = []
            citation_info = []
            random_info = []
            for i in batch_gather_data:
                paper_info.append(i[0])
                citation_info.extend([j for j in i[1:] if "label" in j and j["label"] == "Related Work"])
                # random_info.extend([j for j in i[1:] if "category" in j])
            chunk_info.extend(citation_info)
            # chunk_info.extend(random_info)
            additional_entries_needed = 100 * num - len(chunk_info)
            rel_w_url = []
            for rel_w in chunk_info:
                rel_w_url.append(rel_w["url"])
            for p in paper_info:
                rel_w_url.append(p["url"])
            random.shuffle(random_papers)
            random_cnt = 0
            for selected_entry in random_papers:
                if selected_entry["url"] in rel_w_url:
                    continue
                chunk_info.append(selected_entry)
                random_cnt += 1
                if random_cnt == additional_entries_needed:
                    break
            save_json_data["paper_info"] = paper_info
            save_json_data["chunk_info"] = chunk_info
            write_to_json([save_json_data], sample_save_path)
            cnt += 1


def refine_prun_co_author_graph(root, file_name="graph.pkl"):
    pass


# TODO: download all the paper pdf and extract main context (not only pdf from latest(test) paper) multi-ip distributed
def construct_co_author_graph(root, start_author, node_limit=5):
    save_path = os.path.join(root, "co_author")
    check_path(save_path)

    graph, node_feat, edge_feat = bfs(start_author, node_limit)
    # TODO: pre-save

    save_raw_paper_path = os.path.join(save_path, "raw_papers")
    check_path(save_raw_paper_path)

    client = arxiv.Client()
    new_node_feat = dict()
    for k, v in node_feat.items():
        save_author_raw_paper_path = os.path.join(save_raw_paper_path, k)
        save_author_raw_paper_pdf_path = os.path.join(save_author_raw_paper_path, "pdf")
        save_author_raw_paper_json_path = os.path.join(save_author_raw_paper_path, "json")
        check_path(save_author_raw_paper_path)
        check_path(save_author_raw_paper_pdf_path)
        check_path(save_author_raw_paper_json_path)
        new_paper_list = []
        for paper in v:
            search = arxiv.Search(id_list=[paper['url'].split("/")[-1]], max_results=1)
            try:
                result = next(client.results(search), None)
            except:
                client = arxiv.Client()
                continue
            if result:
                print("Processing Paper: ", result.title)
                paper_info = {
                    'url': result.entry_id,
                    'title': result.title,
                    'abstract': result.summary,
                    'authors': ', '.join(author.name for author in result.authors),
                    'published': str(result.published).split(" ")[0],
                    'updated': str(result.updated).split(" ")[0],
                    'primary_cat': result.primary_category,
                    'cats': result.categories,
                    'label': "Original Paper"
                }
                identifier = result.entry_id.split("/")[-1]
                if os.path.exists(os.path.join(save_author_raw_paper_pdf_path, identifier + '.pdf')) and \
                        os.path.exists(os.path.join(save_author_raw_paper_json_path, identifier + '.json')):
                    new_paper_list.append(paper_info)
                    continue
                TOL = 1
                while TOL:
                    try:
                        result.download_pdf(dirpath=save_author_raw_paper_pdf_path, filename=identifier + '.pdf')
                        break
                    except:
                        time.sleep(5)
                        TOL -= 1
                if TOL == 0:
                    print("Skip")
                    continue
                write_to_json(paper_info, os.path.join(save_author_raw_paper_json_path, identifier + '.json'))
                new_paper_list.append(paper_info)
        new_node_feat[k] = new_paper_list

    save_data = {"graph": graph, "node_feat": new_node_feat, "edge_feat": edge_feat}
    write_to_json(save_data, os.path.join(save_path, "graph.json"))


def anonymize():
    pass


def chronological_split(root, setting="title_short"):
    files = os.listdir(os.path.join(root, setting))
    split_list = [[], [], []]
    if setting == "title_short":
        val_cutoff = datetime.date(year=2024, month=3, day=25)
        test_cutoff = datetime.date(year=2024, month=4, day=25)
    elif setting == "abstract_short":
        val_cutoff = datetime.date(year=2024, month=3, day=25)
        test_cutoff = datetime.date(year=2024, month=4, day=25)
    elif setting == "introduction_short":
        val_cutoff = datetime.date(year=2024, month=3, day=25)
        test_cutoff = datetime.date(year=2024, month=4, day=25)
    elif setting == "related_short":
        val_cutoff = datetime.date(year=2024, month=3, day=25)
        test_cutoff = datetime.date(year=2024, month=4, day=25)
    else:
        raise Exception("Error")
    for file in files:
        if not file.endswith(".json"):
            continue
        file_path = os.path.join(root, setting, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = json.load(f)
        if setting == "related_short":
            pub_time = sample[0]["published"].split("-")
        else:
            pub_time = sample["published"].split("-")
        pub_date = datetime.date(year=int(pub_time[0]), month=int(pub_time[1]), day=int(pub_time[2]))
        if pub_date >= test_cutoff:
            split_list[2].append(file)
        elif pub_date >= val_cutoff:
            split_list[1].append(file)
        else:
            split_list[0].append(file)

    total = len(split_list[0]) + len(split_list[1]) + len(split_list[2])
    print("AcademicEval {}: Train/Val/Test: {}/{}/{}=={}/{}/{}".format(setting, len(split_list[0]), len(split_list[1]),
                                                                       len(split_list[2]), len(split_list[0]) / total,
                                                                       len(split_list[1]) / total,
                                                                       len(split_list[2]) / total))
    with open(os.path.join(root, setting, "train.txt"), "w", encoding="utf-8") as f:
        for i in split_list[0]:
            f.write(i)
            f.write('\n')
    with open(os.path.join(root, setting, "val.txt"), "w", encoding="utf-8") as f:
        for i in split_list[1]:
            f.write(i)
            f.write('\n')
    with open(os.path.join(root, setting, "test.txt"), "w", encoding="utf-8") as f:
        for i in split_list[2]:
            f.write(i)
            f.write('\n')


def stats_related_multi_task(data):
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased")
    tokenizer.max_len = 99999
    with open(data, 'r', encoding='utf-8') as f:
        sample = json.load(f)
    out_len = len(tokenizer(sample[0]['gt'])['input_ids'])
    in_str = ""
    for i in sample:
        in_str += i["title"] + " " + i["abstract"]
    in_len = len(tokenizer(in_str)['input_ids'])

    return in_len, out_len


def stats_related_multi(root):
    files = os.listdir(os.path.join(root, "related_multi"))
    with Pool(40) as p:
        ret = p.map(stats_related_multi_task, [os.path.join(root, "related_multi", i) for i in files if i.endswith(".json")])

    in_len = []
    out_len = []
    for sub_in_len, sub_out_len in ret:
        in_len.append(sub_in_len)
        out_len.append(sub_out_len)

    print("Related Multi: in_len/out_len=={}/{}".format(np.mean(in_len), np.mean(out_len)))


def stats_abstract_single_task(data):
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased")
    tokenizer.max_len = 99999
    with open(data, 'r', encoding='utf-8') as f:
        sample = json.load(f)
    out_len = len(tokenizer(sample['gt'])['input_ids'])
    in_len = len(tokenizer(sample['main_content'])['input_ids'])

    return in_len, out_len


def stats_abstract_single(root):
    files = os.listdir(os.path.join(root, "abstract_single"))
    with Pool(40) as p:
        ret = p.map(stats_abstract_single_task, [os.path.join(root, "abstract_single", i) for i in files if i.endswith(".json")])

    in_len = []
    out_len = []
    for sub_in_len, sub_out_len in ret:
        in_len.append(sub_in_len)
        out_len.append(sub_out_len)

    print("Abstract Single: in_len/out_len=={}/{}".format(np.mean(in_len), np.mean(out_len)))


def stats_abstract_multi_task(data):
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased")
    tokenizer.max_len = 99999
    with open(data, 'r', encoding='utf-8') as f:
        sample = json.load(f)
    out_len = 0
    for sub_gt in sample[0]['gt']:
        out_len += len(tokenizer(sub_gt)['input_ids'])
    out_len /= 10
    in_str = ""
    for i in sample[1:]:
        in_str += i["main_content"] + " "
    in_len = len(tokenizer(in_str)['input_ids'])

    return in_len, out_len


def stats_abstract_multi(root):
    files = os.listdir(os.path.join(root, "abstract_multi"))
    with Pool(40) as p:
        ret = p.map(stats_abstract_multi_task, [os.path.join(root, "abstract_multi", i) for i in files if i.endswith(".json")])

    in_len = []
    out_len = []
    for sub_in_len, sub_out_len in ret:
        in_len.append(sub_in_len)
        out_len.append(sub_out_len)

    print("Abstract Multi: in_len/out_len=={}/{}".format(np.mean(in_len), np.mean(out_len)))


def stats_title_single_task(data):
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased")
    tokenizer.max_len = 99999
    with open(data, 'r', encoding='utf-8') as f:
        sample = json.load(f)
    out_len = len(tokenizer(sample['gt'])['input_ids'])
    in_len = len(tokenizer(sample['main_content'])['input_ids'])

    return in_len, out_len


def stats_title_single(root):
    files = os.listdir(os.path.join(root, "title_single"))
    with Pool(40) as p:
        ret = p.map(stats_title_single_task, [os.path.join(root, "title_single", i) for i in files if i.endswith(".json")])

    in_len = []
    out_len = []
    for sub_in_len, sub_out_len in ret:
        in_len.append(sub_in_len)
        out_len.append(sub_out_len)

    print("Title Single: in_len/out_len=={}/{}".format(np.mean(in_len), np.mean(out_len)))


def stats_short_task(data):
    file_path, setting = data
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased")
    tokenizer.max_len = 99999
    with open(file_path, 'r', encoding='utf-8') as f:
        sample = json.load(f)
    if not isinstance(sample, list):
        out_len = len(tokenizer(sample['gt'])['input_ids'])
        if setting == "title":
            in_len = len(tokenizer(sample['main_content'] + " " + sample['abstract'])['input_ids'])
        elif setting == "abstract":
            in_len = len(tokenizer(sample['main_content'] + " " + sample['title'])['input_ids'])
        elif setting == "introduction":
            in_len = len(tokenizer(sample['main_content'] + " " + sample['title'] + " " + sample['abstract'])['input_ids'])
        else:
            raise Exception("Error")
    else:
        out_len = len(tokenizer(sample[0]['gt'])['input_ids'])
        in_str = sample[0]['main_content'] + " " + sample[0]['title'] + " " + sample[0]['abstract']
        for i in sample[1:]:
            in_str += i["title"] + " " + i["abstract"]
        in_len = len(tokenizer(in_str)['input_ids'])

    return in_len, out_len


def stats_short(root, setting):
    files = os.listdir(os.path.join(root, "{}_short".format(setting)))
    with Pool(40) as p:
        ret = p.map(stats_short_task, [(os.path.join(root, "{}_short".format(setting), i), setting) for i in files if i.endswith(".json")])

    in_len = []
    out_len = []
    for sub_in_len, sub_out_len in ret:
        in_len.append(sub_in_len)
        out_len.append(sub_out_len)

    print("{} Short: in_len/out_len=={}/{}".format(setting.title(), np.mean(in_len), np.mean(out_len)))


def stats_long_task(data):
    MAX_NUM = 1
    # MAX_NUM = 2
    file_path, setting, augmented_with_graph, both = data
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased")
    tokenizer.max_len = 99999
    with open(file_path, 'r', encoding='utf-8') as f:
        sample = json.load(f)
    if not isinstance(sample, list):
        out_len = len(tokenizer(sample['gt'])['input_ids'])
        if setting == "title":
            in_str = sample['main_content'] + " " + sample['abstract']
            if augmented_with_graph:
                cnt = 0
                stop = False
                for _, add_info in sample["additional_graph_info"]["node_feat"].items():
                    for sub_add_info in add_info:
                        if sub_add_info["url"].find(file_path.split("_")[-1].replace(".json", "")) != -1:
                            continue
                        in_str += sub_add_info['main_content'] + " " + sub_add_info['abstract'] + " " + sub_add_info["title"]
                        cnt += 1
                        if cnt >= MAX_NUM:
                            stop = True
                            break
                    if stop:
                        break
                if both:
                    for add_info in sample["additional_info"][:MAX_NUM]:
                        in_str += add_info['main_content'] + " " + add_info['abstract'] + " " + add_info["gt"]
            else:
                for add_info in sample["additional_info"][:MAX_NUM]:
                    in_str += add_info['main_content'] + " " + add_info['abstract'] + " " + add_info["gt"]
        elif setting == "abstract":
            in_str = sample['main_content'] + " " + sample['title']
            if augmented_with_graph:
                cnt = 0
                stop = False
                for _, add_info in sample["additional_graph_info"]["node_feat"].items():
                    for sub_add_info in add_info:
                        if sub_add_info["url"].find(file_path.split("_")[-1].replace(".json", "")) != -1:
                            continue
                        in_str += sub_add_info['main_content'] + " " + sub_add_info['title'] + " " + sub_add_info["abstract"]
                        cnt += 1
                        if cnt >= MAX_NUM:
                            stop = True
                            break
                    if stop:
                        break
                if both:
                    for add_info in sample["additional_info"][:MAX_NUM]:
                        in_str += add_info['main_content'] + " " + add_info['title'] + " " + add_info["gt"]
            else:
                for add_info in sample["additional_info"][:MAX_NUM]:
                    in_str += add_info['main_content'] + " " + add_info['title'] + " " + add_info["gt"]
        elif setting == "introduction":
            in_str = sample['main_content'] + " " + sample['title'] + " " + sample['abstract']
            if augmented_with_graph:
                cnt = 0
                stop = False
                for _, add_info in sample["additional_graph_info"]["node_feat"].items():
                    for sub_add_info in add_info:
                        if sub_add_info["url"].find(file_path.split("_")[-1].replace(".json", "")) != -1:
                            continue
                        in_str += sub_add_info['main_content'] + " " + sub_add_info['title'] + " " + sub_add_info["abstract"] + " " + sub_add_info["introduction"]
                        cnt += 1
                        if cnt >= MAX_NUM:
                            stop = True
                            break
                    if stop:
                        break
                if both:
                    for add_info in sample["additional_info"][:MAX_NUM]:
                        in_str += add_info['main_content'] + " " + add_info['title'] + " " + add_info['abstract'] + " " + add_info["gt"]
            else:
                for add_info in sample["additional_info"][:MAX_NUM]:
                    in_str += add_info['main_content'] + " " + add_info['title'] + " " + add_info['abstract'] + " " + add_info["gt"]
        else:
            raise Exception("Error")
        in_len = len(tokenizer(in_str)['input_ids'])
    else:
        out_len = len(tokenizer(sample[0]['gt'])['input_ids'])
        in_str = sample[0]['main_content'] + " " + sample[0]['title'] + " " + sample[0]['abstract']
        for i in sample[1:]:
            in_str += i["title"] + " " + i["abstract"]
        if augmented_with_graph:
            cnt = 0
            stop = False
            for _, add_info in sample[0]["additional_graph_info"]["node_feat"].items():
                for sub_add_info in add_info:
                    if sub_add_info["url"].find(file_path.split("_")[-1].replace(".json", "")) != -1:
                        continue
                    in_str += sub_add_info['main_content'] + " " + sub_add_info['title'] + " " + sub_add_info["abstract"] + " " + sub_add_info["related"]
                    cnt += 1
                    if cnt >= MAX_NUM:
                        stop = True
                        break
                if stop:
                    break
            if both:
                for add_info in sample[0]["additional_info"][:MAX_NUM]:
                    in_str += add_info[0]['main_content'] + " " + add_info[0]['title'] + " " + add_info[0]['abstract'] + " " + add_info[0]['gt']
        else:
            for add_info in sample[0]["additional_info"][:MAX_NUM]:
                in_str += add_info[0]['main_content'] + " " + add_info[0]['title'] + " " + add_info[0]['abstract'] + " " + add_info[0]['gt']
        in_len = len(tokenizer(in_str)['input_ids'])

    return in_len, out_len


def stats_long(root, setting, augmented_with_graph=False, both=False):
    if augmented_with_graph:
        if not both:
            files = os.listdir(os.path.join(root, "{}_long_graph".format(setting)))
            with Pool(40) as p:
                ret = p.map(stats_long_task,
                            [(os.path.join(root, "{}_long_graph".format(setting), i), setting, augmented_with_graph, both) for i in
                             files if i.endswith(".json")])
        else:
            files = os.listdir(os.path.join(root, "{}_long_mix".format(setting)))
            with Pool(40) as p:
                ret = p.map(stats_long_task,
                            [(os.path.join(root, "{}_long_mix".format(setting), i), setting, augmented_with_graph,
                              both) for i in
                             files if i.endswith(".json")])
    else:
        files = os.listdir(os.path.join(root, "{}_long".format(setting)))
        with Pool(40) as p:
            ret = p.map(stats_long_task, [(os.path.join(root, "{}_long".format(setting), i), setting, augmented_with_graph, both) for i in files if i.endswith(".json")])

    in_len = []
    out_len = []
    for sub_in_len, sub_out_len in ret:
        in_len.append(sub_in_len)
        out_len.append(sub_out_len)

    print("{} Long: in_len/out_len=={}/{}".format(setting.title(), np.mean(in_len), np.mean(out_len)))


def reverse_filter(root, setting):
    check_path(os.path.join(root, "{}_short_old".format(setting)))
    check_path(os.path.join(root, "{}_long_old".format(setting)))
    standard = [i.split("_")[-1] for i in os.listdir(os.path.join(root, "{}_long_graph".format(setting)))]
    old = [i.split("_")[-1] for i in os.listdir(os.path.join(root, "{}_long".format(setting)))]
    remove_files = list(set(old) - set(standard))
    for i in remove_files:
        shutil.move(os.path.join(root, "{}_short".format(setting), "{}_short_{}".format(setting, i)), os.path.join(root, "{}_short_old".format(setting)))
        shutil.move(os.path.join(root, "{}_long".format(setting), "{}_long_{}".format(setting, i)), os.path.join(root, "{}_long_old".format(setting)))

    with open(os.path.join(root, "{}_short".format(setting), "test.txt"), "r", encoding="utf-8") as f:
        split_files = f.readlines()
    split_files = [i.strip().split("_")[-1] for i in split_files]
    remove_entries = set(split_files) - set(standard)
    new_split_files = []
    for i in split_files:
        if i not in remove_entries:
            new_split_files.append(i)

    with open(os.path.join(root, "{}_short".format(setting), "test.txt"), "w", encoding="utf-8") as f:
        for i in new_split_files:
            f.write(i)
            f.write('\n')

    with open(os.path.join(root, "{}_short".format(setting), "val.txt"), "r", encoding="utf-8") as f:
        split_files = f.readlines()
    split_files = [i.strip().split("_")[-1] for i in split_files]
    remove_entries = set(split_files) - set(standard)
    new_split_files = []
    for i in split_files:
        if i not in remove_entries:
            new_split_files.append(i)

    with open(os.path.join(root, "{}_short".format(setting), "val.txt"), "w", encoding="utf-8") as f:
        for i in new_split_files:
            f.write(i)
            f.write('\n')

    with open(os.path.join(root, "{}_short".format(setting), "train.txt"), "r", encoding="utf-8") as f:
        split_files = f.readlines()
    split_files = [i.strip().split("_")[-1] for i in split_files]
    remove_entries = set(split_files) - set(standard)
    new_split_files = []
    for i in split_files:
        if i not in remove_entries:
            new_split_files.append(i)

    with open(os.path.join(root, "{}_short".format(setting), "train.txt"), "w", encoding="utf-8") as f:
        for i in new_split_files:
            f.write(i)
            f.write('\n')


def split_count(root, setting):
    with open(os.path.join(root, "{}_short".format(setting), "train.txt"), "r", encoding="utf-8") as f:
        train_files = f.readlines()
    with open(os.path.join(root, "{}_short".format(setting), "val.txt"), "r", encoding="utf-8") as f:
        val_files = f.readlines()
    with open(os.path.join(root, "{}_short".format(setting), "test.txt"), "r", encoding="utf-8") as f:
        test_files = f.readlines()

    total = len(train_files) + len(val_files) + len(test_files)
    print(total)
    print(len(train_files) / total, len(val_files) / total, len(test_files) / total)


if __name__ == '__main__':
    pass
