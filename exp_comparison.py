import os
import json
import argparse
import datetime
from tqdm import tqdm
from bert_score import score
from bart_score import BARTScorer
from rouge_score import rouge_scorer
import textstat
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import *
from refine_graph import get_neighbors_within_n_hops, get_adj_list
from retrieval import *


def start_prompt_instruction(setting, target_only=False):
    if setting.find("related") != -1:
        if target_only:
            return "Please read the following Target Content and Target Citations carefully and summarize " \
                   "the Target Citations according to the topic of the Target Content as required.\n"
        else:
            return "Please read the following Reference Content and Output carefully and summarize " \
                   "the Target Citations according to the topic of the Target Content as required.\n"
    else:
        if target_only:
            return "Please read the following Target Content carefully and summarize the Target Content as required.\n"
        else:
            return "Please read the following Reference Content and Output carefully and summarize the Target Content as required.\n"


# TODO: Prompt & Few-shot Examples
def prompt_pool(setting, rag):
    if setting.find("abstract") != -1:
        if rag:
            prompt = "Please craft an abstract summarizing the key points from all the above provided Target Contents. " \
                     "The abstract should be of appropriate length (around 200 words) and include the main theme, significant findings or arguments, " \
                     "and conclusions of the Target Contents. Please ensure that the abstract captures the essence " \
                     "of the Target Contents in a clear, coherent, and succinct manner. " \
                     "Please output the abstract directly without including other redundant or irrelevant text."
        else:
            if setting.find("short") != -1:
                prompt = "Please craft an abstract summarizing the key points from the above provided Target Content. " \
                         "The abstract should be of appropriate length (around 200 words) and include the main theme, significant findings or arguments, " \
                         "and conclusions of the Target Content. Please ensure that the abstract captures the essence " \
                         "of the Target Content in a clear, coherent, and succinct manner. " \
                         "Please output the abstract directly without including other redundant or irrelevant text."
            else:
                prompt = "Please craft an abstract summarizing the key points from the above provided Target Content. " \
                         "The Reference Content and Output provide some demonstrations, which may also contain some information that " \
                         "is potentially related to the Target Content. You can refer to the input and output text forms of " \
                         "the Reference Content and Output to assist in summarizing the Target Content, and try to explore and use the " \
                         "information that is potentially related to the Target Content contained in the Reference Content and Output." \
                         "The abstract should be of appropriate length (around 200 words) and include the main theme, significant findings or arguments, " \
                         "and conclusions of the Target Content. Please ensure that the abstract captures the essence " \
                         "of the Target Content in a clear, coherent, and succinct manner. " \
                         "Please output the abstract directly without including other redundant or irrelevant text."
    elif setting.find("title") != -1:
        if rag:
            prompt = "Please craft a title highly summarizing the main theme from all the above provided Target Contents. " \
                     "The title should be of appropriate length (strictly limited to about 10 words). " \
                     "The title should also include and highlight the core and most critical theme " \
                     "of the Target Contents, ignoring minor and redundant information. " \
                     "Please ensure that the title captures the essence of the Target Contents in a clear and concise manner. " \
                     "Please output the title directly without including other redundant or irrelevant text."
        else:
            if setting.find("short") != -1:
                prompt = "Please craft a title highly summarizing the main theme from the above provided Target Content. " \
                         "The title should be of appropriate length (strictly limited to about 10 words). " \
                         "The title should also include and highlight the core and most critical theme " \
                         "of the Target Content, ignoring minor and redundant information. " \
                         "Please ensure that the title captures the essence of the Target Content in a clear and concise manner. " \
                         "Please output the title directly without including other redundant or irrelevant text."
            else:
                prompt = "Please craft a title highly summarizing the main theme from the above provided Target Content. " \
                         "The Reference Content and Output provide some demonstrations, which may also contain some information that " \
                         "is potentially related to the Target Content. You can refer to the input and output text forms of " \
                         "the Reference Content and Output to assist in summarizing the Target Content, and try to explore and use the " \
                         "information that is potentially related to the Target Content contained in the Reference Content and Output." \
                         "The title should be of appropriate length (strictly limited to about 10 words). " \
                         "The title should also include and highlight the core and most critical theme " \
                         "of the Target Content, ignoring minor and redundant information. " \
                         "Please ensure that the title captures the essence of the Target Content in a clear and concise manner. " \
                         "Please output the title directly without including other redundant or irrelevant text."
    elif setting.find("introduction") != -1:
        if rag:
            prompt = "Please craft an introduction summarizing the key points from all the above provided Target Contents. " \
                     "The introduction should be of appropriate length (about 1000 to 1500 words). " \
                     "The introduction should first describe the topic or main theme of the Target Contents, then provide relevant " \
                     "background knowledge, and summarize the existing relevant research on this topic from the Target Contents, " \
                     "point out their advantages and disadvantages, and highly summarize the specific research problem and " \
                     "problem statement targeted by the Target Contents. Next, describe in detail the core approach or " \
                     "insights proposed by the Target Contents on this topic, and include any necessary experimental results." \
                     "Then, use about 3 short paragraphs (each paragraph is about 50 words) to highly summarize the " \
                     "approach or insights proposed in the Target Contents, as well as the experimental results. " \
                     "Finally, briefly give an overview of the Target Contents' structure. " \
                     "Please ensure that the introduction captures the essence of the Target Contents in a clear, coherent, and succinct manner. " \
                     "Please output the introduction directly without including other redundant or irrelevant text."
        else:
            if setting.find("short") != -1:
                prompt = "Please craft an introduction summarizing the key points from the above provided Target Content. " \
                         "The introduction should be of appropriate length (about 1000 to 1500 words). " \
                         "The introduction should first describe the topic or main theme of the Target Content, then provide relevant " \
                         "background knowledge, and summarize the existing relevant research on this topic from the Target Content, " \
                         "point out their advantages and disadvantages, and highly summarize the specific research problem and " \
                         "problem statement targeted by the Target Content. Next, describe in detail the core approach or " \
                         "insights proposed by the Target Content on this topic, and include any necessary experimental results." \
                         "Then, use about 3 short paragraphs (each paragraph is about 50 words) to highly summarize the " \
                         "approach or insights proposed in the Target Content, as well as the experimental results. " \
                         "Finally, briefly give an overview of the Target Content's structure. " \
                         "Please ensure that the introduction captures the essence of the Target Content in a clear, coherent, and succinct manner. " \
                         "Please output the introduction directly without including other redundant or irrelevant text."
            else:
                prompt = "Please craft an introduction summarizing the key points from the above provided Target Content. " \
                         "The Reference Content and Output provide some demonstrations, which may also contain some information that " \
                         "is potentially related to the Target Content. You can refer to the input and output text forms of " \
                         "the Reference Content and Output to assist in summarizing the Target Content, and try to explore and use the " \
                         "information that is potentially related to the Target Content contained in the Reference Content and Output." \
                         "The introduction should be of appropriate length (about 1000 to 1500 words). " \
                         "The introduction should first describe the topic or main theme of the Target Content, then provide relevant " \
                         "background knowledge, and summarize the existing relevant research on this topic from the Target Content, " \
                         "point out their advantages and disadvantages, and highly summarize the specific research problem and " \
                         "problem statement targeted by the Target Content. Next, describe in detail the core approach or " \
                         "insights proposed by the Target Content on this topic, and include any necessary experimental results." \
                         "Then, use about 3 short paragraphs (each paragraph is about 50 words) to highly summarize the " \
                         "approach or insights proposed in the Target Content, as well as the experimental results. " \
                         "Finally, briefly give an overview of the Target Content's structure. " \
                         "Please ensure that the introduction captures the essence of the Target Content in a clear, coherent, and succinct manner. " \
                         "Please output the introduction directly without including other redundant or irrelevant text."
    elif setting.find("related") != -1:
        if rag:
            prompt = "Given the Target Content Abstract and Title, " \
                     "please craft a related work summarizing the key points from all the above provided Target Contents. " \
                     "There is no specific length requirement or limit for the entire related work (it is best to keep it around 500 to 1000 words), but each Target Content that " \
                     "appears in the related work needs to be highly summarized in extremely concise and short sentences." \
                     "You can refer to the topic or main theme described by the Target Content Abstract and Title to " \
                     "filter irrelevant information in the Target Contents and leverage relevant information. " \
                     "Furthermore, you can categorize the relevant Target Contents and briefly summarize the advantages " \
                     "and disadvantages of each categorization. Please ensure that the related work captures all " \
                     "the relevant key points of the Target Contents in a clear, coherent, and succinct manner. " \
                     "Please output the related work directly without including other redundant or irrelevant text."
        else:
            if setting.find("short") != -1:
                prompt = "Given the Target Content and its Abstract and Title, along with its Target Citations (including Target Citation Title and Abstract), " \
                         "please craft a related work summarizing the key points from the above provided Target Citations. " \
                         "There is no specific length requirement or limit for the entire related work (it is best to keep it around 500 to 1000 words), but each Target Citation that " \
                         "appears in the related work needs to be highly summarized in extremely concise and short sentences." \
                         "You can refer to the topic or main theme described by the Target Content and its Abstract and Title to " \
                         "filter irrelevant information in the Target Citations and leverage relevant information. " \
                         "Furthermore, you can categorize the relevant Target Citations, briefly summarize the advantages " \
                         "and disadvantages of each categorization, and explain the advantages of the approach proposed " \
                         "in the Target Content. Please ensure that the related work captures all the relevant " \
                         "key points of the Target Citations in a clear, coherent, and succinct manner. " \
                         "Please output the related work directly without including other redundant or irrelevant text."
            else:
                prompt = "Given the Target Content and its Abstract and Title, along with its Target Citations (including Target Citation Title and Abstract), " \
                         "please craft a related work summarizing the key points from the above provided Target Citations. " \
                         "The Reference Content and Output provide some demonstrations, which may also contain some information that " \
                         "is potentially related to the Target Content. You can refer to the input and output text forms of " \
                         "the Reference Content and Output to assist in summarizing the Target Citations, and try to explore and use the " \
                         "information (e.g., related citations missing from the Target Citations) that is potentially " \
                         "related to the Target Content contained in the Reference Content and Output." \
                         "There is no specific length requirement or limit for the entire related work (it is best to keep it around 500 to 1000 words), but each Target Citation that " \
                         "appears in the related work needs to be highly summarized in extremely concise and short sentences." \
                         "You can refer to the topic or main theme described by the Target Content and its Abstract and Title to " \
                         "filter irrelevant information in the Target Citations and leverage relevant information. " \
                         "Furthermore, you can categorize the relevant Target Citations, briefly summarize the advantages " \
                         "and disadvantages of each categorization, and explain the advantages of the approach proposed " \
                         "in the Target Content. Please ensure that the related work captures all the relevant " \
                         "key points of the Target Citations in a clear, coherent, and succinct manner. " \
                         "Please output the related work directly without including other redundant or irrelevant text."
    else:
        raise Exception("Setting Not Exist")

    return prompt


# TODO: Prompt & Few-shot Examples
def rag_query_prompt(setting, info):
    if setting.find("abstract") != -1:
        prompt = "Please craft an abstract summarizing the key points of the provided text. The title of the text is: {}".format(info)
    elif setting.find("title") != -1:
        prompt = "Please craft a title highly summarizing the main theme of the provided text. The abstract of the text is: {}".format(info)
    elif setting.find("introduction") != -1:
        prompt = "Please craft an introduction summarizing the main theme of the provided text " \
                 "(including background knowledge, advantages and disadvantages of existing research and challenges, " \
                 "the proposed approach, experimental results, etc.). " \
                 "The title of the text is: {}. The abstract of the text is: {}.".format(info["title"], info["abstract"])
    elif setting.find("related") != -1:
        prompt = "Please craft a related work summarizing all the relevant key points of the provided text. " \
                 "The title of the text is: {}. The abstract of the text is: {}.".format(info["title"], info["abstract"])
    else:
        raise Exception("Setting Not Exist")

    return prompt


# def bart_score_eval(generate_response, ground_truth):
#     if not isinstance(ground_truth[0], str):
#         num_ref = len(ground_truth[0])
#         generate_response_expand = []
#         ground_truth_expand = []
#         for i, j in zip(generate_response, ground_truth):
#             generate_response_expand.extend([i] * num_ref)
#             ground_truth_expand.extend(j)
#
#         bart_scorer = BARTScorer(device=DEVICE, checkpoint='bart-large-cnn')
#         res_1 = bart_scorer.score(generate_response_expand, ground_truth_expand, batch_size=8)
#         bart_scorer.load(path='bart_score.pth')
#         res_2 = bart_scorer.score(generate_response_expand, ground_truth_expand, batch_size=8)
#
#         res_1_narrow = []
#         res_2_narrow = []
#         for i in range(len(generate_response)):
#             res_1_narrow.append(max(res_1[i * num_ref: (i + 1) * num_ref]))
#             res_2_narrow.append(max(res_2[i * num_ref: (i + 1) * num_ref]))
#
#         return [float(i) for i in res_1_narrow], [float(i) for i in res_2_narrow]
#     else:
#         bart_scorer = BARTScorer(device=DEVICE, checkpoint='bart-large-cnn')
#         res_1 = bart_scorer.score(generate_response, ground_truth, batch_size=8)
#         bart_scorer.load(path='bart_score.pth')
#         res_2 = bart_scorer.score(generate_response, ground_truth, batch_size=8)
#
#         return [float(i) for i in res_1], [float(i) for i in res_2]


def bert_score_eval(generate_response, ground_truth, opt=0):
    if opt == 0:
        P, R, F = score(generate_response, ground_truth, lang="en", device=DEVICE, batch_size=8)
    else:
        P, R, F = score(generate_response, ground_truth, model_type="microsoft/deberta-xlarge-mnli", device=DEVICE, batch_size=8)
    P = [float(i) for i in P.numpy()]
    R = [float(i) for i in R.numpy()]
    F = [float(i) for i in F.numpy()]

    return P, R, F


def rouge_eval(generate_response, ground_truth, type='rougeL'):
    if not isinstance(ground_truth, str):
        num_ref = len(ground_truth)
        generate_response_expand = [generate_response] * num_ref
        ground_truth_expand = ground_truth
        Ps = []
        Rs = []
        Fs = []
        for i, j in zip(generate_response_expand, ground_truth_expand):
            scorer = rouge_scorer.RougeScorer([type], use_stemmer=True)
            scores = scorer.score(prediction=i, target=j)
            Ps.append(scores[type].precision)
            Rs.append(scores[type].recall)
            Fs.append(scores[type].fmeasure)
        P = max(Ps)
        R = max(Rs)
        F = max(Fs)

        return float(P), float(R), float(F)
    else:
        scorer = rouge_scorer.RougeScorer([type], use_stemmer=True)
        scores = scorer.score(prediction=generate_response, target=ground_truth)
        P = scores[type].precision
        R = scores[type].recall
        F = scores[type].fmeasure

        return float(P), float(R), float(F)


def response_eval(generate_responses, ground_truthes, file_identifiers):
    result_recorder = dict()
    metric_list = []
    bert_Ps, bert_Rs, bert_Fs = bert_score_eval(generate_responses, ground_truthes, opt=0)
    bert_ms_Ps, bert_ms_Rs, bert_ms_Fs = bert_score_eval(generate_responses, ground_truthes, opt=1)
    # bart_res_1s, bart_res_2s = bart_score_eval(generate_responses, ground_truthes)

    for ind, (generate_response, ground_truth, file_identifier) in enumerate(tqdm(zip(generate_responses, ground_truthes, file_identifiers))):
        metrics = dict()
        rouge_L_P, rouge_L_R, rouge_L_F = rouge_eval(generate_response, ground_truth, type='rougeL')
        rouge_1_P, rouge_1_R, rouge_1_F = rouge_eval(generate_response, ground_truth, type='rouge1')
        rouge_2_P, rouge_2_R, rouge_2_F = rouge_eval(generate_response, ground_truth, type='rouge2')

        metrics["BERT SCORE"] = {"P": bert_Ps[ind], "R": bert_Rs[ind], "F": bert_Fs[ind]}
        metrics["BERT SCORE MS"] = {"P": bert_ms_Ps[ind], "R": bert_ms_Rs[ind], "F": bert_ms_Fs[ind]}
        # metrics["BART SCORE"] = {"RES1": bart_res_1s[ind], "RES2": bart_res_2s[ind]}
        metrics["ROUGE-L"] = {"P": rouge_L_P, "R": rouge_L_R, "F": rouge_L_F}
        metrics["ROUGE-1"] = {"P": rouge_1_P, "R": rouge_1_R, "F": rouge_1_F}
        metrics["ROUGE-2"] = {"P": rouge_2_P, "R": rouge_2_R, "F": rouge_2_F}
        metrics["Readability"] = {"Flesch-Kincaid": textstat.flesch_reading_ease(generate_response),
                                  "ARI": textstat.automated_readability_index(generate_response)}

        result_recorder[file_identifier] = {"gt": ground_truth, "response": generate_response, "metrics": metrics}
        metric_list.append(metrics)

    all_metrics = dict()
    for key in metric_list[0].keys():
        all_metrics[key] = {kk: float(np.mean([vv[key][kk] for vv in metric_list])) for kk in
                            metric_list[0][key].keys()}

    print("\n")
    print(text_wrap("=" * 50 + "Final Evaluation" + "=" * 50))
    print_metrics(all_metrics)

    result_recorder["all_metrics"] = all_metrics

    return result_recorder


def trunc(text, max_len):
    # TODO: Accurate Truncation
    # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    #     model_name="gpt-4",
    #     chunk_size=max_len,
    #     chunk_overlap=0,
    # )
    # return text_splitter.split_text(text)[0]

    ids = TOKENIZER.encode(text, max_length=max_len, truncation=True)

    return TOKENIZER.decode(ids, skip_special_tokens=True)


def get_token_len(text):
    return len(TOKENIZER(text)['input_ids'])


def rag_retrieval(corpus, rag_query, require_split=True):
    if require_split:
        chunk_list = TEXT_SPLITTER.split_text(corpus)
    else:
        chunk_list = corpus
    if len(chunk_list) <= RECALL_CHUNK_NUM:
        return chunk_list
    if RETRIEVER not in ["bm25", "tf-idf"]:
        chunk_embedding = get_dense_embedding(chunk_list, retriever=RETRIEVER, tokenizer=CTX_TOKENIZER,
                                              model=CTX_ENCODER)
        rag_query_embedding = get_dense_embedding(rag_query, retriever=RETRIEVER, tokenizer=QUERY_TOKENIZER,
                                                  model=QUERY_ENCODER)
        _, retrieved_text_list = run_dense_retrieval(rag_query_embedding, chunk_embedding, chunk_list,
                                                     chunk_num=RECALL_CHUNK_NUM)
    else:
        chunk_embedding = get_sparse_retriever(chunk_list, retriever=RETRIEVER, num=RECALL_CHUNK_NUM)
        _, retrieved_text_list = run_sparse_retrieval(rag_query, chunk_embedding, chunk_list)

    return retrieved_text_list


def get_graph_info_list(sample):
    info_list = []
    cnt = 0
    stop = False
    for _, add_info in sample["additional_graph_info"]["node_feat"].items():
        for sub_add_info in add_info:
            if sub_add_info["url"] == sample["url"]:
                continue
            info_dict = {"main_content": sub_add_info['main_content'], "abstract": sub_add_info['abstract'],
                         "title": sub_add_info["title"]}
            if "introduction" in sub_add_info:
                info_dict["introduction"] = sub_add_info["introduction"]
            if "related" in sub_add_info:
                info_dict["related"] = sub_add_info["related"]
            info_list.append(info_dict)
            cnt += 1
            if cnt >= MAX_NUM:
                stop = True
                break
        if stop:
            break

    return info_list


def eval_title_short(sample, llm_model, rag):
    sample["main_content"] = sample["main_content"].replace(sample["title"], "")
    gt = sample["gt"]
    prompt = start_prompt_instruction(setting="title_short", target_only=True)
    if rag:
        rag_query = rag_query_prompt(setting="title_short", info=sample["abstract"])
        retrieved_text_list = rag_retrieval(corpus=sample["main_content"], rag_query=rag_query)
        for ind, retrieved_text in enumerate(retrieved_text_list):
            prompt += "### Target Content {}: {}\n".format(ind, retrieved_text)
    else:
        prompt += "### Target Content: {}\n".format(sample["main_content"])

    pre_total_len = get_token_len(prompt +
                                  "### Target Content Abstract: {}\n".format(sample["abstract"]) +
                                  prompt_pool(setting="title_short", rag=rag))
    diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - 32)
    if diff_len > 0:
        prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
    prompt += "### Target Content Abstract: {}\n".format(sample["abstract"])
    prompt += prompt_pool(setting="title_short", rag=rag)
    response = get_llm_response_via_api(prompt=prompt,
                                        LLM_MODEL=llm_model,
                                        TAU=TAU,
                                        SEED=SEED)
    print(response)

    return response.replace("Sure, here is the title", ""), gt


def eval_title_long(sample, llm_model, rag):
    sample["main_content"] = sample["main_content"].replace(sample["title"], "")
    gt = sample["gt"]
    prompt = start_prompt_instruction(setting="title_long", target_only=True if rag else False)
    if rag:
        rag_query = rag_query_prompt(setting="title_long", info=sample["abstract"])
        in_corpus = sample["main_content"]
        for i in sample["additional_info"][:MAX_NUM]:
            in_corpus += i['main_content'] + " " + i['abstract'] + " " + i["gt"]
        retrieved_text_list = rag_retrieval(corpus=in_corpus, rag_query=rag_query)
        for ind, retrieved_text in enumerate(retrieved_text_list):
            prompt += "### Target Content {}: {}\n".format(ind, retrieved_text)
        pre_total_len = get_token_len(prompt +
                                      "### Target Content Abstract: {}\n".format(sample["abstract"]) +
                                      prompt_pool(setting="title_long", rag=rag))
        diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - 32)
        if diff_len > 0:
            prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
    else:
        for ind, i in enumerate(sample["additional_info"][:MAX_NUM]):
            prompt += "### Reference Content {}: {}\n ### Reference Abstract {}: {}\n ### Reference Output {}: {}\n".format(ind, i['main_content'], ind, i['abstract'], ind, i["gt"])
        pre_total_len = get_token_len(prompt +
                                      "### Target Content: {}\n".format(sample["main_content"]) +
                                      "### Target Content Abstract: {}\n".format(sample["abstract"]) +
                                      prompt_pool(setting="title_long", rag=rag))
        diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - 32)
        if diff_len > 0:
            if get_token_len(prompt) - diff_len > 0:
                prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
                prompt += "### Target Content: {}\n".format(sample["main_content"])
            else:
                prompt = start_prompt_instruction(setting="title_long", target_only=True if rag else False)
                prompt += "### Target Content: {}\n".format(sample["main_content"])
                pre_total_len = get_token_len(prompt +
                                              "### Target Content Abstract: {}\n".format(sample["abstract"]) +
                                              prompt_pool(setting="title_long", rag=rag))
                diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - 32)
                if diff_len > 0:
                    prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
        else:
            prompt += "### Target Content: {}\n".format(sample["main_content"])


    prompt += "### Target Content Abstract: {}\n".format(sample["abstract"])
    prompt += prompt_pool(setting="title_long", rag=rag)
    response = get_llm_response_via_api(prompt=prompt,
                                        LLM_MODEL=llm_model,
                                        TAU=TAU,
                                        SEED=SEED)
    print(response)

    return response.replace("Sure, here is the title", ""), gt


def eval_title_long_graph(sample, llm_model, rag):
    sample["main_content"] = sample["main_content"].replace(sample["title"], "")
    gt = sample["gt"]
    prompt = start_prompt_instruction(setting="title_long_graph", target_only=True if rag else False)
    if rag:
        rag_query = rag_query_prompt(setting="title_long_graph", info=sample["abstract"])
        in_corpus = sample["main_content"]
        for i in get_graph_info_list(sample):
            in_corpus += i['main_content'] + " " + i['abstract'] + " " + i["title"]
        retrieved_text_list = rag_retrieval(corpus=in_corpus, rag_query=rag_query)
        for ind, retrieved_text in enumerate(retrieved_text_list):
            prompt += "### Target Content {}: {}\n".format(ind, retrieved_text)
        pre_total_len = get_token_len(prompt +
                                      "### Target Content Abstract: {}\n".format(sample["abstract"]) +
                                      prompt_pool(setting="title_long_graph", rag=rag))
        diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - 32)
        if diff_len > 0:
            prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
    else:
        for ind, i in enumerate(get_graph_info_list(sample)):
            prompt += "### Reference Content {}: {}\n ### Reference Abstract {}: {}\n ### Reference Output {}: {}\n".format(
                ind, i['main_content'], ind, i['abstract'], ind, i["title"])
        pre_total_len = get_token_len(prompt +
                                      "### Target Content: {}\n".format(sample["main_content"]) +
                                      "### Target Content Abstract: {}\n".format(sample["abstract"]) +
                                      prompt_pool(setting="title_long_graph", rag=rag))
        diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - 32)
        if diff_len > 0:
            if get_token_len(prompt) - diff_len > 0:
                prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
                prompt += "### Target Content: {}\n".format(sample["main_content"])
            else:
                prompt = start_prompt_instruction(setting="title_long_graph", target_only=True if rag else False)
                prompt += "### Target Content: {}\n".format(sample["main_content"])
                pre_total_len = get_token_len(prompt +
                                              "### Target Content Abstract: {}\n".format(sample["abstract"]) +
                                              prompt_pool(setting="title_long_graph", rag=rag))
                diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - 32)
                if diff_len > 0:
                    prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
        else:
            prompt += "### Target Content: {}\n".format(sample["main_content"])

    prompt += "### Target Content Abstract: {}\n".format(sample["abstract"])
    prompt += prompt_pool(setting="title_long_graph", rag=rag)
    response = get_llm_response_via_api(prompt=prompt,
                                        LLM_MODEL=llm_model,
                                        TAU=TAU,
                                        SEED=SEED)
    print(response)

    return response.replace("Sure, here is the title", ""), gt


def eval_abstract_short(sample, llm_model, rag):
    gt = sample["gt"]
    prompt = start_prompt_instruction(setting="abstract_short", target_only=True)
    if rag:
        rag_query = rag_query_prompt(setting="abstract_short", info=sample["title"])
        retrieved_text_list = rag_retrieval(corpus=sample["main_content"], rag_query=rag_query)
        for ind, retrieved_text in enumerate(retrieved_text_list):
            prompt += "### Target Content {}: {}\n".format(ind, retrieved_text)
    else:
        prompt += "### Target Content: {}\n".format(sample["main_content"])

    pre_total_len = get_token_len(prompt +
                                  "### Target Content Title: {}\n".format(sample["title"]) +
                                  prompt_pool(setting="abstract_short", rag=rag))
    diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - (32 + 512))
    if diff_len > 0:
        prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
    prompt += "### Target Content Title: {}\n".format(sample["title"])
    prompt += prompt_pool(setting="abstract_short", rag=rag)
    response = get_llm_response_via_api(prompt=prompt,
                                        LLM_MODEL=llm_model,
                                        TAU=TAU,
                                        SEED=SEED)
    print(response)

    return response, gt


def eval_abstract_long(sample, llm_model, rag):
    gt = sample["gt"]
    prompt = start_prompt_instruction(setting="abstract_long", target_only=True if rag else False)
    if rag:
        rag_query = rag_query_prompt(setting="abstract_long", info=sample["title"])
        in_corpus = sample["main_content"]
        for i in sample["additional_info"][:MAX_NUM]:
            in_corpus += i['main_content'] + " " + i['title'] + " " + i["gt"]
        retrieved_text_list = rag_retrieval(corpus=in_corpus, rag_query=rag_query)
        for ind, retrieved_text in enumerate(retrieved_text_list):
            prompt += "### Target Content {}: {}\n".format(ind, retrieved_text)
        pre_total_len = get_token_len(prompt +
                                      "### Target Content Title: {}\n".format(sample["title"]) +
                                      prompt_pool(setting="abstract_long", rag=rag))
        diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - (32 + 512))
        if diff_len > 0:
            prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
    else:
        for ind, i in enumerate(sample["additional_info"][:MAX_NUM]):
            prompt += "### Reference Content {}: {}\n ### Reference Title {}: {}\n ### Reference Output {}: {}\n".format(
                ind, i['main_content'], ind, i['title'], ind, i["gt"])
        pre_total_len = get_token_len(prompt +
                                      "### Target Content: {}\n".format(sample["main_content"]) +
                                      "### Target Content Title: {}\n".format(sample["title"]) +
                                      prompt_pool(setting="abstract_long", rag=rag))
        diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - (32 + 512))
        if diff_len > 0:
            if get_token_len(prompt) - diff_len > 0:
                prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
                prompt += "### Target Content: {}\n".format(sample["main_content"])
            else:
                prompt = start_prompt_instruction(setting="abstract_long", target_only=True if rag else False)
                prompt += "### Target Content: {}\n".format(sample["main_content"])
                pre_total_len = get_token_len(prompt +
                                              "### Target Content Title: {}\n".format(sample["title"]) +
                                              prompt_pool(setting="abstract_long", rag=rag))
                diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - (32 + 512))
                if diff_len > 0:
                    prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
        else:
            prompt += "### Target Content: {}\n".format(sample["main_content"])

    prompt += "### Target Content Title: {}\n".format(sample["title"])
    prompt += prompt_pool(setting="abstract_long", rag=rag)
    response = get_llm_response_via_api(prompt=prompt,
                                        LLM_MODEL=llm_model,
                                        TAU=TAU,
                                        SEED=SEED)
    print(response)

    return response, gt


def eval_abstract_long_graph(sample, llm_model, rag):
    gt = sample["gt"]
    prompt = start_prompt_instruction(setting="abstract_long_graph", target_only=True if rag else False)
    if rag:
        rag_query = rag_query_prompt(setting="abstract_long_graph", info=sample["title"])
        in_corpus = sample["main_content"]
        for i in get_graph_info_list(sample):
            in_corpus += i['main_content'] + " " + i['title'] + " " + i["abstract"]
        retrieved_text_list = rag_retrieval(corpus=in_corpus, rag_query=rag_query)
        for ind, retrieved_text in enumerate(retrieved_text_list):
            prompt += "### Target Content {}: {}\n".format(ind, retrieved_text)
        pre_total_len = get_token_len(prompt +
                                      "### Target Content Title: {}\n".format(sample["title"]) +
                                      prompt_pool(setting="abstract_long_graph", rag=rag))
        diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - (32 + 512))
        if diff_len > 0:
            prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
    else:
        for ind, i in enumerate(get_graph_info_list(sample)):
            prompt += "### Reference Content {}: {}\n ### Reference Title {}: {}\n ### Reference Output {}: {}\n".format(
                ind, i['main_content'], ind, i['title'], ind, i["abstract"])
        pre_total_len = get_token_len(prompt +
                                      "### Target Content: {}\n".format(sample["main_content"]) +
                                      "### Target Content Title: {}\n".format(sample["title"]) +
                                      prompt_pool(setting="abstract_long_graph", rag=rag))
        diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - (32 + 512))
        if diff_len > 0:
            if get_token_len(prompt) - diff_len > 0:
                prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
                prompt += "### Target Content: {}\n".format(sample["main_content"])
            else:
                prompt = start_prompt_instruction(setting="abstract_long_graph", target_only=True if rag else False)
                prompt += "### Target Content: {}\n".format(sample["main_content"])
                pre_total_len = get_token_len(prompt +
                                              "### Target Content Title: {}\n".format(sample["title"]) +
                                              prompt_pool(setting="abstract_long_graph", rag=rag))
                diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - (32 + 512))
                if diff_len > 0:
                    prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
        else:
            prompt += "### Target Content: {}\n".format(sample["main_content"])

    prompt += "### Target Content Title: {}\n".format(sample["title"])
    prompt += prompt_pool(setting="abstract_long_graph", rag=rag)
    response = get_llm_response_via_api(prompt=prompt,
                                        LLM_MODEL=llm_model,
                                        TAU=TAU,
                                        SEED=SEED)
    print(response)

    return response, gt


def eval_introduction_short(sample, llm_model, rag):
    gt = sample["gt"]
    prompt = start_prompt_instruction(setting="introduction_short", target_only=True)
    if rag:
        rag_query = rag_query_prompt(setting="introduction_short", info={"title": sample["title"], "abstract": sample["abstract"]})
        retrieved_text_list = rag_retrieval(corpus=sample["main_content"], rag_query=rag_query)
        for ind, retrieved_text in enumerate(retrieved_text_list):
            prompt += "### Target Content {}: {}\n".format(ind, retrieved_text)
    else:
        prompt += "### Target Content: {}\n".format(sample["main_content"])

    pre_total_len = get_token_len(prompt +
                                  "### Target Content Title: {}\n".format(sample["title"]) +
                                  "### Target Content Abstract: {}\n".format(sample["abstract"]) +
                                  prompt_pool(setting="introduction_short", rag=rag))
    diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - (32 + 1536))
    if diff_len > 0:
        prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
    prompt += "### Target Content Title: {}\n".format(sample["title"])
    prompt += "### Target Content Abstract: {}\n".format(sample["abstract"])
    prompt += prompt_pool(setting="introduction_short", rag=rag)
    response = get_llm_response_via_api(prompt=prompt,
                                        LLM_MODEL=llm_model,
                                        TAU=TAU,
                                        SEED=SEED)
    print(response)

    return response, gt


def eval_introduction_long(sample, llm_model, rag):
    gt = sample["gt"]
    prompt = start_prompt_instruction(setting="introduction_long", target_only=True if rag else False)
    if rag:
        rag_query = rag_query_prompt(setting="introduction_long", info={"title": sample["title"], "abstract": sample["abstract"]})
        in_corpus = sample["main_content"]
        for i in sample["additional_info"][:MAX_NUM]:
            in_corpus += i['main_content'] + " " + i['title'] + " " + i["abstract"] + " " + i["gt"]
        retrieved_text_list = rag_retrieval(corpus=in_corpus, rag_query=rag_query)
        for ind, retrieved_text in enumerate(retrieved_text_list):
            prompt += "### Target Content {}: {}\n".format(ind, retrieved_text)
        pre_total_len = get_token_len(prompt +
                                      "### Target Content Title: {}\n".format(sample["title"]) +
                                      "### Target Content Abstract: {}\n".format(sample["abstract"]) + 
                                      prompt_pool(setting="introduction_long", rag=rag))
        diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - (32 + 1536))
        if diff_len > 0:
            prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
    else:
        for ind, i in enumerate(sample["additional_info"][:MAX_NUM]):
            prompt += "### Reference Content {}: {}\n ### Reference Title {}: {}\n ### Reference Abstract {}: {}\n ### Reference Output {}: {}\n".format(
                ind, i['main_content'], ind, i['title'], ind, i["abstract"], ind, i["gt"])
        pre_total_len = get_token_len(prompt +
                                      "### Target Content: {}\n".format(sample["main_content"]) +
                                      "### Target Content Title: {}\n".format(sample["title"]) +
                                      "### Target Content Abstract: {}\n".format(sample["abstract"]) +
                                      prompt_pool(setting="introduction_long", rag=rag))
        diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - (32 + 1536))
        if diff_len > 0:
            if get_token_len(prompt) - diff_len > 0:
                prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
                prompt += "### Target Content: {}\n".format(sample["main_content"])
            else:
                prompt = start_prompt_instruction(setting="introduction_long", target_only=True if rag else False)
                prompt += "### Target Content: {}\n".format(sample["main_content"])
                pre_total_len = get_token_len(prompt +
                                              "### Target Content Title: {}\n".format(sample["title"]) +
                                              "### Target Content Abstract: {}\n".format(sample["abstract"]) + 
                                              prompt_pool(setting="introduction_long", rag=rag))
                diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - (32 + 1536))
                if diff_len > 0:
                    prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
        else:
            prompt += "### Target Content: {}\n".format(sample["main_content"])

    prompt += "### Target Content Title: {}\n".format(sample["title"])
    prompt += "### Target Content Abstract: {}\n".format(sample["abstract"])
    prompt += prompt_pool(setting="introduction_long", rag=rag)
    response = get_llm_response_via_api(prompt=prompt,
                                        LLM_MODEL=llm_model,
                                        TAU=TAU,
                                        SEED=SEED)
    print(response)

    return response, gt


def eval_introduction_long_graph(sample, llm_model, rag):
    gt = sample["gt"]
    prompt = start_prompt_instruction(setting="introduction_long_graph", target_only=True if rag else False)
    if rag:
        rag_query = rag_query_prompt(setting="introduction_long_graph",
                                     info={"title": sample["title"], "abstract": sample["abstract"]})
        in_corpus = sample["main_content"]
        for i in get_graph_info_list(sample):
            in_corpus += i['main_content'] + " " + i['title'] + " " + i["abstract"] + " " + i["introduction"]
        retrieved_text_list = rag_retrieval(corpus=in_corpus, rag_query=rag_query)
        for ind, retrieved_text in enumerate(retrieved_text_list):
            prompt += "### Target Content {}: {}\n".format(ind, retrieved_text)
        pre_total_len = get_token_len(prompt +
                                      "### Target Content Title: {}\n".format(sample["title"]) +
                                      "### Target Content Abstract: {}\n".format(sample["abstract"]) +
                                      prompt_pool(setting="introduction_long_graph", rag=rag))
        diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - (32 + 1536))
        if diff_len > 0:
            prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
    else:
        for ind, i in enumerate(get_graph_info_list(sample)):
            prompt += "### Reference Content {}: {}\n ### Reference Title {}: {}\n ### Reference Abstract {}: {}\n ### Reference Output {}: {}\n".format(
                ind, i['main_content'], ind, i['title'], ind, i["abstract"], ind, i["introduction"])
        pre_total_len = get_token_len(prompt +
                                      "### Target Content: {}\n".format(sample["main_content"]) +
                                      "### Target Content Title: {}\n".format(sample["title"]) +
                                      "### Target Content Abstract: {}\n".format(sample["abstract"]) +
                                      prompt_pool(setting="introduction_long_graph", rag=rag))
        diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - (32 + 1536))
        if diff_len > 0:
            if get_token_len(prompt) - diff_len > 0:
                prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
                prompt += "### Target Content: {}\n".format(sample["main_content"])
            else:
                prompt = start_prompt_instruction(setting="introduction_long_graph", target_only=True if rag else False)
                prompt += "### Target Content: {}\n".format(sample["main_content"])
                pre_total_len = get_token_len(prompt +
                                              "### Target Content Title: {}\n".format(sample["title"]) +
                                              "### Target Content Abstract: {}\n".format(sample["abstract"]) +
                                              prompt_pool(setting="introduction_long_graph", rag=rag))
                diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - (32 + 1536))
                if diff_len > 0:
                    prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
        else:
            prompt += "### Target Content: {}\n".format(sample["main_content"])

    prompt += "### Target Content Title: {}\n".format(sample["title"])
    prompt += "### Target Content Abstract: {}\n".format(sample["abstract"])
    prompt += prompt_pool(setting="introduction_long_graph", rag=rag)
    response = get_llm_response_via_api(prompt=prompt,
                                        LLM_MODEL=llm_model,
                                        TAU=TAU,
                                        SEED=SEED)
    print(response)

    return response, gt


def eval_related_short(sample, llm_model, rag):
    gt = sample[0]["gt"]
    prompt = start_prompt_instruction(setting="related_short", target_only=True)
    if rag:
        rag_query = rag_query_prompt(setting="related_short",
                                     info={"title": sample[0]["title"], "abstract": sample[0]["abstract"]})
        in_corpus = sample[0]["main_content"]
        for i in sample[1:]:
            in_corpus += i["title"] + " " + i["abstract"]
        retrieved_text_list = rag_retrieval(corpus=in_corpus, rag_query=rag_query)
        for ind, retrieved_text in enumerate(retrieved_text_list):
            prompt += "### Target Content {}: {}\n".format(ind, retrieved_text)
    else:
        for ind, i in enumerate(sample[1:]):
            prompt += "### Target Citation {}:\n Target Citation Title: {}\n Target Citation Abstract: {}\n".format(ind, i["title"], i["abstract"])
        prompt += "### Target Content: {}\n".format(sample[0]["main_content"])

    pre_total_len = get_token_len(prompt +
                                  "### Target Content Title: {}\n".format(sample[0]["title"]) +
                                  "### Target Content Abstract: {}\n".format(sample[0]["abstract"]) +
                                  prompt_pool(setting="related_short", rag=rag))
    diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - (32 + 1024))
    if diff_len > 0:
        prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
    prompt += "### Target Content Title: {}\n".format(sample[0]["title"])
    prompt += "### Target Content Abstract: {}\n".format(sample[0]["abstract"])
    prompt += prompt_pool(setting="related_short", rag=rag)
    response = get_llm_response_via_api(prompt=prompt,
                                        LLM_MODEL=llm_model,
                                        TAU=TAU,
                                        SEED=SEED)
    print(response)

    return response, gt


def eval_related_long(sample, llm_model, rag):
    gt = sample[0]["gt"]
    prompt = start_prompt_instruction(setting="related_long", target_only=True if rag else False)
    if rag:
        rag_query = rag_query_prompt(setting="related_long",
                                     info={"title": sample[0]["title"], "abstract": sample[0]["abstract"]})
        in_corpus = sample[0]["main_content"]
        for i in sample[1:]:
            in_corpus += i["title"] + " " + i["abstract"]
        for i in sample[0]["additional_info"][:MAX_NUM]:
            in_corpus += i[0]['main_content'] + " " + i[0]['title'] + " " + i[0]["abstract"] + " " + i[0]["gt"]
        retrieved_text_list = rag_retrieval(corpus=in_corpus, rag_query=rag_query)
        for ind, retrieved_text in enumerate(retrieved_text_list):
            prompt += "### Target Content {}: {}\n".format(ind, retrieved_text)
        pre_total_len = get_token_len(prompt +
                                      "### Target Content Title: {}\n".format(sample[0]["title"]) +
                                      "### Target Content Abstract: {}\n".format(sample[0]["abstract"]) +
                                      prompt_pool(setting="related_long", rag=rag))
        diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - (32 + 1024))
        if diff_len > 0:
            prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
    else:
        for ind, i in enumerate(sample[0]["additional_info"][:MAX_NUM]):
            prompt += "### Reference Content {}: {}\n ### Reference Title {}: {}\n ### Reference Abstract {}: {}\n ### Reference Output {}: {}\n".format(
                ind, i[0]['main_content'], ind, i[0]['title'], ind, i[0]["abstract"], ind, i[0]["gt"])
        pre_total_len = get_token_len(prompt +
                                      "".join(["### Target Citation {}:\n Target Citation Title: {}\n Target Citation Abstract: {}\n".format(ind, i["title"], i["abstract"]) for ind, i in enumerate(sample[1:])]) +
                                      "### Target Content: {}\n".format(sample[0]["main_content"]) +
                                      "### Target Content Title: {}\n".format(sample[0]["title"]) +
                                      "### Target Content Abstract: {}\n".format(sample[0]["abstract"]) +
                                      prompt_pool(setting="related_long", rag=rag))
        diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - (32 + 1024))
        if diff_len > 0:
            if get_token_len(prompt) - diff_len > 0:
                prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
                for ind, i in enumerate(sample[1:]):
                    prompt += "### Target Citation {}:\n Target Citation Title: {}\n Target Citation Abstract: {}\n".format(ind, i["title"], i["abstract"])
                prompt += "### Target Content: {}\n".format(sample[0]["main_content"])
            else:
                prompt = start_prompt_instruction(setting="related_long", target_only=True if rag else False)
                for ind, i in enumerate(sample[1:]):
                    prompt += "### Target Citation {}:\n Target Citation Title: {}\n Target Citation Abstract: {}\n".format(ind, i["title"], i["abstract"])
                prompt += "### Target Content: {}\n".format(sample[0]["main_content"])
                pre_total_len = get_token_len(prompt +
                                              "### Target Content Title: {}\n".format(sample[0]["title"]) +
                                              "### Target Content Abstract: {}\n".format(sample[0]["abstract"]) +
                                              prompt_pool(setting="related_long", rag=rag))
                diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - (32 + 1024))
                if diff_len > 0:
                    prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
        else:
            for ind, i in enumerate(sample[1:]):
                prompt += "### Target Citation {}:\n Target Citation Title: {}\n Target Citation Abstract: {}\n".format(ind, i["title"], i["abstract"])
            prompt += "### Target Content: {}\n".format(sample[0]["main_content"])

    prompt += "### Target Content Title: {}\n".format(sample[0]["title"])
    prompt += "### Target Content Abstract: {}\n".format(sample[0]["abstract"])
    prompt += prompt_pool(setting="related_long", rag=rag)
    response = get_llm_response_via_api(prompt=prompt,
                                        LLM_MODEL=llm_model,
                                        TAU=TAU,
                                        SEED=SEED)
    print(response)

    return response, gt


def eval_related_long_graph(sample, llm_model, rag):
    gt = sample[0]["gt"]
    prompt = start_prompt_instruction(setting="related_long_graph", target_only=True if rag else False)
    if rag:
        rag_query = rag_query_prompt(setting="related_long_graph",
                                     info={"title": sample[0]["title"], "abstract": sample[0]["abstract"]})
        in_corpus = sample[0]["main_content"]
        for i in sample[1:]:
            in_corpus += i["title"] + " " + i["abstract"]
        for i in get_graph_info_list(sample[0]):
            in_corpus += i['main_content'] + " " + i['title'] + " " + i["abstract"] + " " + i["related"]
        retrieved_text_list = rag_retrieval(corpus=in_corpus, rag_query=rag_query)
        for ind, retrieved_text in enumerate(retrieved_text_list):
            prompt += "### Target Content {}: {}\n".format(ind, retrieved_text)
        pre_total_len = get_token_len(prompt +
                                      "### Target Content Title: {}\n".format(sample[0]["title"]) +
                                      "### Target Content Abstract: {}\n".format(sample[0]["abstract"]) +
                                      prompt_pool(setting="related_long_graph", rag=rag))
        diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - (32 + 1024))
        if diff_len > 0:
            prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
    else:
        for ind, i in enumerate(get_graph_info_list(sample[0])):
            prompt += "### Reference Content {}: {}\n ### Reference Title {}: {}\n ### Reference Abstract {}: {}\n ### Reference Output {}: {}\n".format(
                ind, i['main_content'], ind, i['title'], ind, i["abstract"], ind, i["related"])
        pre_total_len = get_token_len(prompt +
                                      "".join(["### Target Citation {}:\n Target Citation Title: {}\n Target Citation Abstract: {}\n".format(ind,
                                                                                                             i["title"],
                                                                                                             i[
                                                                                                                 "abstract"])
                                               for ind, i in enumerate(sample[1:])]) +
                                      "### Target Content: {}\n".format(sample[0]["main_content"]) +
                                      "### Target Content Title: {}\n".format(sample[0]["title"]) +
                                      "### Target Content Abstract: {}\n".format(sample[0]["abstract"]) +
                                      prompt_pool(setting="related_long_graph", rag=rag))
        diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - (32 + 1024))
        if diff_len > 0:
            if get_token_len(prompt) - diff_len > 0:
                prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
                for ind, i in enumerate(sample[1:]):
                    prompt += "### Target Citation {}:\n Target Citation Title: {}\n Target Citation Abstract: {}\n".format(ind, i["title"],
                                                                                            i["abstract"])
                prompt += "### Target Content: {}\n".format(sample[0]["main_content"])
            else:
                prompt = start_prompt_instruction(setting="related_long_graph", target_only=True if rag else False)
                for ind, i in enumerate(sample[1:]):
                    prompt += "### Target Citation {}:\n Target Citation Title: {}\n Target Citation Abstract: {}\n".format(ind, i["title"],
                                                                                            i["abstract"])
                prompt += "### Target Content: {}\n".format(sample[0]["main_content"])
                pre_total_len = get_token_len(prompt +
                                              "### Target Content Title: {}\n".format(sample[0]["title"]) +
                                              "### Target Content Abstract: {}\n".format(sample[0]["abstract"]) +
                                              prompt_pool(setting="related_long_graph", rag=rag))
                diff_len = pre_total_len - (CONTEXT_LEN_MAP[llm_model] - (32 + 1024))
                if diff_len > 0:
                    prompt = trunc(prompt, max_len=get_token_len(prompt) - diff_len)
        else:
            for ind, i in enumerate(sample[1:]):
                prompt += "### Target Citation {}:\n Target Citation Title: {}\n Target Citation Abstract: {}\n".format(ind, i["title"], i["abstract"])
            prompt += "### Target Content: {}\n".format(sample[0]["main_content"])

    prompt += "### Target Content Title: {}\n".format(sample[0]["title"])
    prompt += "### Target Content Abstract: {}\n".format(sample[0]["abstract"])
    prompt += prompt_pool(setting="related_long_graph", rag=rag)
    response = get_llm_response_via_api(prompt=prompt,
                                        LLM_MODEL=llm_model,
                                        TAU=TAU,
                                        SEED=SEED)
    print(response)

    return response, gt


def exp_llm(setting, llm_model, rag):
    root = "./AcademicEval"
    result_recorder = dict()
    check_path("./exp")
    if not rag:
        result_recorder_save_path = "./exp/res_{}_{}_{}_{}_{}_{}.json".format(setting, llm_model.split("/")[-1], RECALL_CHUNK_NUM, CHUNK_SIZE, CHUNK_OVERLAP, MAX_NUM)
    else:
        result_recorder_save_path = "./exp/res_{}_{}_{}_{}_{}_{}_{}.json".format(setting, llm_model.split("/")[-1], RETRIEVER, RECALL_CHUNK_NUM, CHUNK_SIZE, CHUNK_OVERLAP, MAX_NUM)
    with open(result_recorder_save_path, 'r', encoding='utf-8') as f:
        result_recorder = json.load(f)
    generated_responses = []
    ground_truthes = []
    file_identifiers = []
    test_files = [i for i in os.listdir(os.path.join(root, setting)) if i.find("test") != -1 and i.endswith("json")]
    for ind, test_file in enumerate(tqdm(test_files)):
        if ind == 142 or ind == 417:
            continue
        if test_file in result_recorder:
            file_identifiers.append(test_file)
            generated_responses.append(result_recorder[test_file]["response"])
            ground_truthes.append(result_recorder[test_file]["gt"])
            continue
        test_file_path = os.path.join(root, setting, test_file)
        with open(test_file_path, 'r', encoding='utf-8') as f:
            sample = json.load(f)
        if setting == "title_10K":
            response, gt = eval_title_short(sample=sample,
                                            llm_model=llm_model,
                                            rag=rag)
        elif setting == "title_30K":
            response, gt = eval_title_long(sample=sample, 
                                           llm_model=llm_model, 
                                           rag=rag)
        elif setting == "title_31K_G":
            response, gt = eval_title_long_graph(sample=sample, 
                                                 llm_model=llm_model, 
                                                 rag=rag)
        elif setting == "abs_9K":
            response, gt = eval_abstract_short(sample=sample,
                                               llm_model=llm_model,
                                               rag=rag)
        elif setting == "abs_28K":
            response, gt = eval_abstract_long(sample=sample,
                                              llm_model=llm_model,
                                              rag=rag)
        elif setting == "abs_29K_G":
            response, gt = eval_abstract_long_graph(sample=sample,
                                                    llm_model=llm_model,
                                                    rag=rag)
        elif setting == "intro_8K":
            response, gt = eval_introduction_short(sample=sample,
                                                   llm_model=llm_model,
                                                   rag=rag)
        elif setting == "intro_28K":
            response, gt = eval_introduction_long(sample=sample,
                                                  llm_model=llm_model,
                                                  rag=rag)
        elif setting == "intro_28K_G":
            response, gt = eval_introduction_long_graph(sample=sample,
                                                        llm_model=llm_model,
                                                        rag=rag)
        elif setting == "related_34K":
            response, gt = eval_related_short(sample=sample,
                                              llm_model=llm_model,
                                              rag=rag)
        elif setting == "related_53K":
            response, gt = eval_related_long(sample=sample,
                                             llm_model=llm_model,
                                             rag=rag)
        elif setting == "related_53K_G":
            response, gt = eval_related_long_graph(sample=sample,
                                                   llm_model=llm_model,
                                                   rag=rag)
        else:
            raise Exception("Setting Not Exist.")

        result_recorder[test_file] = {"gt": gt, "response": response}
        file_identifiers.append(test_file)
        generated_responses.append(response)
        ground_truthes.append(gt)
        if ind % 10 == 0:
            write_to_json(result_recorder, result_recorder_save_path)
        # if ind == 18:
        #     break

    result_recorder = response_eval(generate_responses=generated_responses,
                                    ground_truthes=ground_truthes,
                                    file_identifiers=file_identifiers)
    result_recorder["status"] = "finished"
    write_to_json(result_recorder, result_recorder_save_path)


def get_context_len_map():
    m = dict()
    m["meta-llama/Llama-3-70b-chat-hf"] = 8192
    m["google/gemma-7b-it"] = 8192
    m["Qwen/Qwen1.5-72B-Chat"] = 32768
    m["mistralai/Mixtral-8x7B-Instruct-v0.1"] = 32768
    m["NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"] = 32768
    m["mistralai/Mixtral-8x22B-Instruct-v0.1"] = 65536

    return m


def get_tokenizer(llm_model):
    if llm_model == "meta-llama/Llama-3-70b-chat-hf":
        return AutoTokenizer.from_pretrained("./llama")
    elif llm_model == "google/gemma-7b-it":
        return AutoTokenizer.from_pretrained("./gemma")
    elif llm_model == "Qwen/Qwen1.5-72B-Chat":
        return AutoTokenizer.from_pretrained("./qwen", trust_remote_code=True)
    elif llm_model == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        return AutoTokenizer.from_pretrained("./mixtral")
    elif llm_model == "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO":
        return AutoTokenizer.from_pretrained("./hermes")
    elif llm_model == "mistralai/Mixtral-8x22B-Instruct-v0.1":
        return AutoTokenizer.from_pretrained("./mixtral")
    else:
        return AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, required=True)
    parser.add_argument("--llm_model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument('--rag', action='store_true')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0)
    parser.add_argument("--retriever", type=str, default="bm25")
    parser.add_argument("--recall_chunk_num", type=int, default=6)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--chunk_overlap", type=int, default=128)
    parser.add_argument("--max_num", type=int, default=2)
    opt = parser.parse_args()
    SETTING = opt.setting
    LLM_MODEL = opt.llm_model
    RAG = opt.rag
    SEED = opt.seed
    TAU = opt.tau
    RETRIEVER = opt.retriever
    RECALL_CHUNK_NUM = opt.recall_chunk_num
    CHUNK_SIZE = opt.chunk_size
    CHUNK_OVERLAP = opt.chunk_overlap
    MAX_NUM = opt.max_num

    set_seed(int(SEED))
    DEVICE = get_device(int(opt.cuda))
    # DEVICE = "cpu"

    CONTEXT_LEN_MAP = get_context_len_map()
    TOKENIZER = get_tokenizer(llm_model=LLM_MODEL)
    TEXT_SPLITTER = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    if RAG and RETRIEVER not in ["bm25", "tf-idf"]:
        QUERY_TOKENIZER, CTX_TOKENIZER, QUERY_ENCODER, CTX_ENCODER = get_dense_retriever(retriever=RETRIEVER)
        QUERY_ENCODER = QUERY_ENCODER.to(DEVICE)
        CTX_ENCODER = CTX_ENCODER.to(DEVICE)

    exp_llm(setting=SETTING,
            llm_model=LLM_MODEL,
            rag=RAG)


