import pickle
import arxiv
import datetime

from construct_relation_graph import arxiv_author_query_format, author_reformat, fetch_author_info


def check_new(author, recall_num=100):
    client = arxiv.Client()
    papers_info = []
    co_authors = []
    new_cnt = 0
    search = arxiv.Search(
        query="{}".format(arxiv_author_query_format(author)),
        max_results=recall_num,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    for ind, result in enumerate(client.results(search)):
        pub_time = str(result.published).split(" ")[0].split("-")
        pub_date = datetime.date(year=int(pub_time[0]), month=int(pub_time[1]), day=int(pub_time[2]))
        if ind == 0 and pub_date < datetime.date(year=2022, month=5, day=25):
            return "outdated", "outdated"
        if pub_date < datetime.date(year=2024, month=5, day=25):
            continue
        author_list = [author_reformat(single_author.name) for single_author in result.authors]
        new_cnt += 1
        for au in author_list:
            if au == author:
                continue
            co_authors.append(au)
        if author == author_list[0]:
            paper_info = {
                'url': result.entry_id,
                "title": result.title,
                "abstract": result.summary,
                "authors": ', '.join(single_author.name for single_author in result.authors),
                "published": str(result.published).split(" ")[0],
                "updated": str(result.updated).split(" ")[0],
                'primary_cat': result.primary_category,
                'cats': result.categories,
            }
            papers_info.append(paper_info)

    co_authors_priority = []
    for co_au in co_authors:
        co_authors_priority.append((co_au, new_cnt))

    return papers_info, co_authors_priority


if __name__ == '__main__':
    with open('./AcademicEval/co_author/graph_refine.pkl', 'rb') as f:
        data = pickle.load(f)

    AUTHOR_LIST = [i for i in data["node_feat"]]
    AUTHOR_SET = set(AUTHOR_LIST)

    CO_AUTHOR_PRIORITY = []
    OUTDATED_AUTHOR = []
    for author in AUTHOR_LIST:
        papers_info, co_author_priority = check_new(author)
        if papers_info == "outdated":
            OUTDATED_AUTHOR.append(author)
        else:
            data["node_feat"][author].extend(papers_info)
            CO_AUTHOR_PRIORITY.extend(co_author_priority)

    CO_AUTHOR_PRIORITY.sort(reverse=True, key=lambda p: p[1])
    for co_author in CO_AUTHOR_PRIORITY:
        papers_info, co_authors = fetch_author_info(co_author)
        for co_au in co_authors:
            if (co_author, co_au) in data["graph"] or (co_au, co_author) in data["graph"]:
                continue
            data["graph"].append((co_author, co_au))

        data["node_feat"][co_author] = papers_info

    remove_eid = []
    for eid, edge in enumerate(data["graph"]):
        a1 = edge[0]
        a2 = edge[1]
        if a1 in OUTDATED_AUTHOR or a2 in OUTDATED_AUTHOR:
            remove_eid.append(eid)

    print(len(data["graph"]))
    [data["graph"].pop(index) for index in sorted(remove_eid, reverse=True)]
    print(len(data["graph"]))

    [data["node_feat"].pop(index) for index in OUTDATED_AUTHOR]

    # go to refine_graph
