import re
import random
from difflib import SequenceMatcher

import fitz
import arxiv
from tqdm import tqdm

from utils import *
from section_region_extractor import detect_region_and_annotate


def search_arxiv_by_title(client, title, tol=10):
    cleaned_title = clean_title(title)
    # Solution 1
    search = arxiv.Search(
        query="ti:{}".format(cleaned_title),
        max_results=tol
    )
    for result in list(client.results(search)):
        if SequenceMatcher(None, cleaned_title.lower(), clean_title(result.title).lower()).ratio() > 0.9:
            return result

    # Solution 2
    search = arxiv.Search(
        query="{}".format(cleaned_title),
        max_results=tol
    )
    for result in list(client.results(search)):
        if SequenceMatcher(None, cleaned_title.lower(), clean_title(result.title).lower()).ratio() > 0.9:
            return result

    return None


def fetch_related_papers_arxiv(papers, file_name):
    fail_cases = []
    client = arxiv.Client()
    with open(file_name, "r", encoding='utf-8') as infile:
        papers_info = json.load(infile)

    found_titles_count = 0
    for title in tqdm(papers, desc="Fetching Related Work Papers", unit="Paper"):
        result = search_arxiv_by_title(client, title, tol=10)
        if result:
            found_titles_count += 1
            paper_info = {
                'url': result.entry_id,
                "title": result.title,
                "abstract": result.summary,
                "authors": ', '.join(author.name for author in result.authors),
                "published": str(result.published).split(" ")[0],
                "updated": str(result.updated).split(" ")[0],
                'primary_cat': result.primary_category,
                'cats': result.categories,
                "label": "Related Work"
            }
            papers_info.append(paper_info)
        else:
            fail_cases.append(title)

    write_to_json(papers_info, file_name)

    print(text_wrap("Num of Related Work Papers Found via API:"), "{}/{}".format(found_titles_count, len(papers)))

    return fail_cases


def fetch_related_papers_scholar(papers, file_name):
    pass
    # raise NotImplementedError


def fetch_original_arxiv(cleaned_title, file_name):
    client = arxiv.Client()
    papers_info = []
    # Search for articles matching the title.
    print("{} Fetching Original...".format(show_time()))
    result = search_arxiv_by_title(client, cleaned_title, tol=10)
    if result:
        print("{} Fetching Successful".format(show_time()))
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
        print(json.dumps(paper_info, indent=4))
        papers_info.append(paper_info)
        write_to_json(papers_info, file_name)

        return True
    else:
        print("{} Fetching Failed: {}".format(show_time(), cleaned_title))
        # raise Exception("Failed When Fetching Original Paper")
        return False


def fetch_original_scholar(cleaned_title, file_name):
    raise NotImplementedError


def finish_dataset(file_name, random_paper_path="random_paper_abs.json"):
    with open(file_name, 'r', encoding='utf-8') as file1:
        paper_info = json.load(file1)
    with open(random_paper_path, 'r', encoding='utf-8') as file2:
        random_papers = json.load(file2)

    entry_count = len(paper_info)
    additional_entries_needed = 100 - entry_count + 1
    rel_w_url = []
    for rel_w in paper_info[1:]:
        rel_w_url.append(rel_w["url"])
    # num_to_sample = min(additional_entries_needed, len(random_papers))
    # selected_entries = random.sample(random_papers, num_to_sample)
    random.shuffle(random_papers)
    cnt = 0
    for selected_entry in random_papers:
        if selected_entry["url"] == paper_info[0]["url"] or selected_entry["url"] in rel_w_url:
            continue
        paper_info.append(selected_entry)
        cnt += 1
        if cnt == additional_entries_needed:
            break

    with open(file_name, 'w', encoding='utf-8') as file1:
        json.dump(paper_info, file1, ensure_ascii=False, indent=4)

    print(text_wrap("Finishing Successful:"), file_name)


def type_pattern_match(text, type=0):
    citations = []
    if type == 0:
        pattern = r'\b[A-Z][a-z]+ et al\.,? \(?\d{4}[a-z]?\)?\b'  # et al
        matches = re.findall(pattern, text)
        citations.extend(matches)

        pattern = r'\b[A-Z][a-z]+, [A-Z][a-z]+, and [A-Z][a-z]+ \(?\d{4}[a-z]?\)?\b'  # Li, Han, and Wu 2018
        matches = re.findall(pattern, text)
        citations.extend(matches)

        pattern = r'\b[A-Z][a-z]+ & [A-Z][a-z]+, \(?\d{4}[a-z]?\)?\b'  # Need & Wun, 1970
        matches = re.findall(pattern, text)
        citations.extend(matches)

        pattern = r'\b[A-Z][a-z]+ and [A-Z][a-z]+,? \(?\d{4}[a-z]?\)?\b'  # Maaten and Hinton 2008
        matches = re.findall(pattern, text)
        citations.extend(matches)
    else:
        pattern = r'\[\d+(?:,\s*\d+)*\]'
        matches = re.findall(pattern, text)
        citations.extend(matches)

    return citations


def num_valid_citation_type(single_citation_text, citation_type):
    if citation_type == 0:
        single_citation_text = single_citation_text.strip()
        # print(single_citation_text)
        return [single_citation_text], 1
    else:
        single_citation_text = single_citation_text[1: -1]
        single_citation_text = single_citation_text.split(",")
        single_citation_text = [i.strip() for i in single_citation_text]
        # print(single_citation_text)
        return single_citation_text, len(single_citation_text)


def citation_count(text, citation_type):
    cnt = 0
    valid_citations = []
    if citation_type == 0:
        matches = type_pattern_match(text, type=0)
    else:
        matches = type_pattern_match(text, type=1)

    for candidate in matches:
        citations, sub_cnt = num_valid_citation_type(candidate, citation_type)
        cnt += sub_cnt
        valid_citations.extend(citations)

    return valid_citations, cnt


def infer_citation_type(text, scope=5000):
    matches_0 = type_pattern_match(text[:min(len(text), scope)], type=0)
    matches_1 = type_pattern_match(text[:min(len(text), scope)], type=1)
    # print(matches_0)
    # print(matches_1)

    if len(matches_0) > len(matches_1):
        return 0  # (Kipf and Welling 2017)
    else:
        return 1  # [5], [1, 5, 23]


def citation_density_detection(pdf_path, text, w_size=1000, step=100, thre=2, tol=5):
    print(text_wrap("[Related Work, References):"), "~{} Chars".format(str(len(text))))
    citation_type = infer_citation_type(text)
    print(text_wrap("Citation Type:"), citation_type)
    cnt_function = []
    accumulation = 0
    normal_exit = True
    for i in range(0, len(text) - w_size + 1, step):
        sub_text = text[i: i + w_size]
        # print(sub_text)
        _, cnt = citation_count(sub_text, citation_type)
        cnt_function.append(cnt)
        if cnt <= thre:
            accumulation += 1
        else:
            accumulation = 0

        if accumulation >= tol:
            normal_exit = False
            break

    if normal_exit:
        end_point = len(text)
    else:
        end_point = i + w_size // 2

    print(text_wrap("Citation Density:"), cnt_function)
    regions, related_text, exclusive_text = detect_region_and_annotate(pdf_path, save=False)
    related_text = related_text.strip().replace('\n', ' ').replace('- ', '')
    # regions = []
    if len(regions) == 0:
        related_text = text[:end_point]
    # print(related_text[-100:])
    citations, _ = citation_count(related_text, citation_type)
    citations = list(set(citations))
    # print(citations)

    return citations, related_text


def extract_related_and_ref(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = chr(12).join([page.get_text() for page in doc])

    abstract_start_idx = max(full_text.find('\nAbstract\n'), full_text.find('\nABSTRACT\n'))
    intro_start_idx = max(full_text.find('Introduction\n', abstract_start_idx), full_text.find('INTRODUCTION\n', abstract_start_idx))
    ref_start_idx = max(full_text.find('References\n', intro_start_idx), full_text.find('REFERENCES\n', intro_start_idx))
    related_choice_1 = max(full_text.find('Related Work\n', intro_start_idx), full_text.find('Related Works\n', intro_start_idx),
                           full_text.find('Related work\n', intro_start_idx), full_text.find('Related works\n', intro_start_idx),
                           full_text.find('RELATED WORK\n', intro_start_idx), full_text.find('RELATED WORKS\n', intro_start_idx),
                           full_text.find('Background and related work\n', intro_start_idx), full_text.find('Background and related works\n', intro_start_idx))
    related_chioce_2 = max(full_text.find('Background\n', intro_start_idx), full_text.find('BACKGROUND\n', intro_start_idx))
    related_start_idx = related_choice_1 if related_choice_1 != -1 else related_chioce_2

    citations, related_text = citation_density_detection(pdf_path, full_text[related_start_idx: ref_start_idx].strip().replace('\n', ' ').replace('- ', ''))
    print(text_wrap("Num of Distinct Citations:"), len(citations))
    print(citations)
    refs = full_text[ref_start_idx:].strip()
    print(text_wrap("\n[References, EOF):"), "~{} Chars".format(str(len(refs))))

    return citations, refs, related_text


def infer_ref_subtype(text):
    for i in text:
        if i.isalpha() or i == '-':
            continue
        if i.isspace():
            return 0  # ACL
        else:
            return 1  # Others


def refine_refs(ref):
    new_ref = ref.strip().replace('\n', ' ').replace('- ', '')
    if new_ref[-1] != ".":
        ind = new_ref.rfind(".")
        new_ref = new_ref[:ind + 1]

    # print(new_ref, "\n")

    return new_ref


def split_refs(refs):
    refs = refs.replace("References", "").replace("REFERENCES", "").strip()
    refs_list = []
    if refs[:100].find("[1]") != -1:
        refs = refs.strip().replace('\n', ' ').replace('- ', '')
        matches = re.finditer(r'\[\d+\]', refs)
        first_ref = next(matches)
        prev = first_ref.end()
        prev_index = refs[first_ref.start(): first_ref.end()].strip()[1:-1]
        for match in matches:
            if str(int(refs[match.start(): match.end()].strip()[1:-1]) -1) != prev_index:
                break
            refs_list.append(refine_refs(refs[prev: match.start()]))
            prev = match.end()
            prev_index = refs[match.start(): match.end()].strip()[1:-1]
    else:
        sub_type = infer_ref_subtype(refs[:50])
        if sub_type == 0:  # ACL
            matches = re.finditer(r'\.\n[A-Z][a-z]+ [A-Z][a-z]+', refs)
            matches = [m for m in matches if refs[m.start() + 2: m.start() + 5] != 'In ']
        else:  # Others
            matches = re.finditer(r'\.\n[A-Z][a-z]+,', refs)

        prev = 0
        for match in matches:
            refs_list.append(refine_refs(refs[prev: match.start() + 1]))
            prev = match.start() + 2

    if len(refs) - prev <= 1000:
        refs_list.append(refine_refs(refs[prev:]))
    else:
        refs_list.append(refine_refs(refs[prev: prev + 500 if prev + 500 <= len(refs) else len(refs)]))

    print(text_wrap("Num of Reference Entries:"), len(refs_list))

    return refs_list



def extract_refs_info(refs):
    new_refs = []
    for ref in refs:
        try:
            cleaned_reference = re.sub(r"[A-Z]\.(?!\s\d)", "", ref)
            cleaned_reference = re.sub(r"\b\d+–\d+\b", "", cleaned_reference).split('.')
            cleaned_reference = [cr for cr in cleaned_reference if len(cr.strip()) != 0]
            first_author = cleaned_reference[0].split(',')[0].strip()
            # print(len(cleaned_reference), cleaned_reference)
            # print(first_author)
            if len(cleaned_reference[1].strip()) <= 5 and cleaned_reference[1].strip()[:4].isdigit():
                year = cleaned_reference[1].strip()
                title = cleaned_reference[2].strip()
                # print(year, title)
            else:
                title = cleaned_reference[1].strip()
                year = cleaned_reference[-1].strip().split(",")[-1].strip()
                # print(year, title)
        except:
            print(text_wrap("Reference Process Error:"), ref)
            continue

        new_ref = dict()
        new_ref['first_author'] = first_author
        new_ref['year'] = year
        new_ref['title'] = title
        new_refs.append(new_ref)

    print(text_wrap("Num of Valid Reference Entries:"), len(new_refs))

    return new_refs


def get_related_titles(citations, refs):
    related_titles = []
    try:
        if citations[0].isdigit():
            for c in citations:
                ref_title = refs[int(c) - 1]['title']
                related_titles.append(ref_title)
        else:
            for c in citations:
                c = c.replace(",", "").split(" ")
                cite_author = c[0].strip()
                cite_year = c[-1].strip()
                if cite_year[0] == '(':
                    cite_year = cite_year[1:]

                candidate = []
                # TODO: Hash
                for ref in refs:
                    if cite_author.lower().strip() in ref['first_author'].lower().strip() and cite_year.lower().strip() == ref['year'].lower().strip():
                        candidate.append(ref)

                if len(candidate) == 0:
                    continue
                else:
                    for can in candidate:
                        related_titles.append(can['title'])
    except Exception as e:
        print(text_wrap("References & Citations Matching Error: {}".format(e)))

    print(text_wrap("Num of Matched Related Work Titles:"), len(related_titles))
    print(related_titles)

    return related_titles



if __name__ == '__main__':
    pass
