import os
import time
import pickle
import arxiv
from tqdm import tqdm

from utils import *


def author_reformat(author):
    return author.strip().title()


def arxiv_author_query_format(author):
    ret = ""
    decompose_author = author.split(" ")
    for ind, part in enumerate(decompose_author):
        ret += "au:" + part
        if ind != len(decompose_author) - 1:
            ret += " AND "

    return ret


def author_position(author, author_list):
    for ind, i in enumerate(author_list):
        if author.lower() == i.lower():
            return ind + 1

    return "NULL"


def co_author_frequency(author, author_list, co_authors):
    for ind, i in enumerate(author_list):
        if author == i:
            continue
        if i in co_authors:
            co_authors[i] += 1
        else:
            co_authors[i] = 1

    return co_authors


def co_author_filter(co_authors, limit=5):
    co_author_list = []
    for k, v in co_authors.items():
        co_author_list.append([k, v])

    co_author_list.sort(reverse=True, key=lambda p: p[1])
    co_author_list = co_author_list[:limit]
    co_author_list = [c[0] for c in co_author_list]

    return co_author_list


def fetch_author_info(author, recall_num=100, max_co_author=20):
    client = arxiv.Client()
    papers_info = []
    co_authors = dict()
    # print("{} Fetching Author Info: {}".format(show_time(), author))
    search = arxiv.Search(
        query="{}".format(arxiv_author_query_format(author)),
        max_results=recall_num,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    for result in client.results(search):
        author_list = [author_reformat(single_author.name) for single_author in result.authors]
        if author != author_list[0]:
            continue
        # author_pos = author_position(author, author_list)
        co_authors = co_author_frequency(author, author_list, co_authors)
        paper_info = {
            'url': result.entry_id,
            "title": result.title,
            "abstract": result.summary,
            "authors": ', '.join(single_author.name for single_author in result.authors),
            "published": str(result.published).split(" ")[0],
            "updated": str(result.updated).split(" ")[0],
            'primary_cat': result.primary_category,
            'cats': result.categories,
            # "author_pos": author_pos
        }
        # print(json.dumps(paper_info, indent=4))
        papers_info.append(paper_info)

    if len(papers_info) == 0:
        raise Exception("No First-author Papers")
    # papers_info.sort(reverse=False, key=lambda p: p["author_pos"])
    co_authors = co_author_filter(co_authors, limit=max_co_author)  # TODO: Topic/...Deepwalk...degree/stark
    # print(text_wrap("Num of Papers:"), len(papers_info))
    # print(text_wrap("Num of Co-authors:"), len(co_authors))

    return papers_info, co_authors


def bfs(ROOT, author_list, save_interval=10, node_limit=999):
    SAVE_PATH = os.path.join(ROOT, "co_author")
    check_path(SAVE_PATH)
    graph = []
    node_feat = dict()
    edge_feat = dict()
    visit = set()
    period_1 = 0
    period_2 = 0
    period_3 = 0
    period_4 = 0
    period_5 = 0
    author_2024 = 0
    for ind, author in enumerate(author_list):
        if author in visit:
            continue
        try:
            papers_info, co_authors = fetch_author_info(author)
        except Exception as e:
            print(text_wrap(e))
            visit.add(author)
            continue
        if len(author_list) <= node_limit:
            author_list.extend(co_authors)
            for co_au in co_authors:
                if (author, co_au) in graph or (co_au, author) in graph:
                    continue
                graph.append((author, co_au))

        visit.add(author)
        node_feat[author] = papers_info
        if ind % save_interval == 0:
            s = time.time()
            save_data = {"graph": graph, "node_feat": node_feat, "edge_feat": edge_feat}
            save_state = {"visit": visit, "author_list": author_list, "ind": ind, "author": author,
                          "save_interval": save_interval, "period_1": period_1, "period_2": period_2,
                          "period_3": period_3, "period_4": period_4, "period_5": period_5, "author_2024": author_2024}
            write_to_pkl(save_data, os.path.join(SAVE_PATH, "graph_tmp.pkl"))
            write_to_pkl(save_state, os.path.join(SAVE_PATH, "graph_state_tmp.pkl"))
            save_time = time.time() - s
            print(save_time)
            if save_time > 1.0:
                save_interval = 100
        try:
            if int(papers_info[0]["published"].split("-")[0]) == 2024:
                author_2024 += 1
        except:
            pass
        for s_paper in papers_info:
            try:
                pub = s_paper["published"].split("-")
                pub_year = int(pub[0])
                pub_month = int(pub[1])
                if pub_year != 2024:
                    continue
                if pub_month == 1:
                    period_1 += 1
                elif pub_month == 2:
                    period_2 += 1
                elif pub_month == 3:
                    period_3 += 1
                elif pub_month == 4:
                    period_4 += 1
                elif pub_month == 5:
                    period_5 += 1
            except:
                continue
        print(text_wrap("Num of Authors: {}/{}, Num of Edges: {}, Author 2024: {}, Period Slot: {}/{}/{}/{}/{}".format(len(node_feat),
                                                                                                      len(author_list),
                                                                                                      len(graph),
                                                                                                      author_2024,
                                                                                                      period_1,
                                                                                                      period_2,
                                                                                                      period_3,
                                                                                                      period_4,
                                                                                                      period_5)))

    save_data = {"graph": graph, "node_feat": node_feat, "edge_feat": edge_feat}
    write_to_pkl(save_data, os.path.join(SAVE_PATH, "graph.pkl"))

    return graph, node_feat, edge_feat


if __name__ == '__main__':
    ROOT = "./AcademicEval"
    start_author = ["YOUR START AUTHOR"]
    bfs(ROOT, start_author)
