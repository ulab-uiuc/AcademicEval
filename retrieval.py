import numpy as np
import torch
import torch.nn.functional as F
import faiss

from transformers import AutoTokenizer, AutoModel
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_community.retrievers.tfidf import TFIDFRetriever
from langchain_core.documents import Document

from utils import show_time


def get_dense_retriever(retriever):
    if retriever == 'contriever':
        query_tokenizer = ctx_tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        query_encoder = ctx_encoder = AutoModel.from_pretrained('facebook/contriever')
    elif retriever == 'dpr':
        query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        query_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        ctx_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    elif retriever == 'dragon':
        query_tokenizer = AutoTokenizer.from_pretrained('facebook/dragon-plus-query-encoder')
        query_encoder = AutoModel.from_pretrained('facebook/dragon-plus-query-encoder')
        ctx_tokenizer = AutoTokenizer.from_pretrained('facebook/dragon-plus-context-encoder')
        ctx_encoder = AutoModel.from_pretrained('facebook/dragon-plus-context-encoder')
    else:
        raise Exception("Error")

    return query_tokenizer, ctx_tokenizer, query_encoder, ctx_encoder


def split_batch(instructions, batch_size):
    batch_instructions = []
    sub_batch = []
    for ind, ins in enumerate(instructions):
        if ind != 0 and ind % batch_size == 0:
            batch_instructions.append(sub_batch)
            sub_batch = [ins]
        else:
            sub_batch.append(ins)

    if len(sub_batch) != 0:
        batch_instructions.append(sub_batch)

    return batch_instructions


def get_dense_embedding(instructions, retriever, tokenizer, model, trunc_len=512, batch_size=64):
    emb_list = []
    batch_instructions = split_batch(instructions, batch_size=batch_size)
    for sub_batch in batch_instructions:
        if retriever == 'contriever':
            inputs = tokenizer(sub_batch, padding=True, truncation=True, return_tensors='pt', max_length=trunc_len).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            def mean_pooling(token_embeddings, mask):
                token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
                sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
                return sentence_embeddings

            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
            for e in embeddings:
                emb_list.append(e)
        elif retriever == 'dpr':
            encoded_input_all = [
                tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=trunc_len)["input_ids"].to(model.device)
                for
                text in sub_batch]
            for inter in encoded_input_all:
                with torch.no_grad():
                    emb_list.append(model(inter).pooler_output.reshape(-1))
        elif retriever == 'dragon':
            inputs = tokenizer(sub_batch, padding=True, truncation=True, return_tensors='pt', max_length=trunc_len).to(model.device)
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state[:, 0, :]
            for e in embeddings:
                emb_list.append(e)
        else:
            raise Exception("Error")

    # print(len(emb_list), emb_list[0].shape)
    return emb_list


def dense_neiborhood_search(corpus_data, query_data, metric='ip', num=8):
    # print(len(corpus_data), corpus_data[0].shape)
    xq = torch.vstack(query_data).cpu().numpy()
    xb = torch.vstack(corpus_data).cpu().numpy()
    dim = xb.shape[1]
    # print(xq.shape, xb.shape)
    if metric == 'l2':
        index = faiss.IndexFlatL2(dim)
    elif metric == 'ip':
        index = faiss.IndexFlatIP(dim)
        xq = xq.astype('float32')
        xb = xb.astype('float32')
        faiss.normalize_L2(xq)
        faiss.normalize_L2(xb)
    else:
        raise Exception("Index Metric Not Exist")
    index.add(xb)
    D, I = index.search(xq, num)

    return I[0]


def get_sparse_retriever(text_chunks, retriever='bm25', num=8):
    documents = [Document(page_content=text) for text in text_chunks]
    if retriever == 'bm25':
        retriever = BM25Retriever.from_documents(documents, k=num)
    elif retriever == 'tf-idf':
        retriever = TFIDFRetriever.from_documents(documents, k=num)
    else:
        raise Exception("Error")

    return retriever


def sparse_neiborhood_search(retriever, queries, text_chunks):
    results = []
    for query in queries:
        # Retrieve documents (assuming no scores are provided)
        retrieved_docs = retriever.get_relevant_documents(query)
        # Map documents to their original indices
        retrieved_indices = [text_chunks.index(doc.page_content) for doc in retrieved_docs]
        # Append the results (just indices)
        results.append(retrieved_indices)

    return results[0]


def calculate_similarity(tensor_list, input_tensor):
    flattened_list = [t.flatten() for t in tensor_list]
    flattened_tensor = input_tensor.flatten()
    cosine_similarities = [F.cosine_similarity(flattened_tensor.unsqueeze(0), t.unsqueeze(0)) for t in flattened_list]

    return cosine_similarities


def run_dense_retrieval(query_embedding, ch_text_chunk_embed, ch_text_chunk, chunk_num=4):
    print("{} Dense Retrieval...".format(show_time()))
    neib_ini = dense_neiborhood_search(ch_text_chunk_embed, query_embedding, num=chunk_num)
    neib_ini = list(neib_ini)

    print("{} Retrieved Chunks:".format(show_time()), neib_ini)
    retrieve_text = []
    for inter3 in neib_ini:
        retrieve_text.append(ch_text_chunk[inter3])

    return neib_ini, retrieve_text


def run_sparse_retrieval(query, ch_text_chunk_embed, text_chunk_l):
    print("{} Sparse Retrieval...".format(show_time()))
    neib_ini = sparse_neiborhood_search(ch_text_chunk_embed, query, text_chunk_l)
    neib = np.array(neib_ini)
    retrieve_text = []
    for inter3 in neib:
        retrieve_text.append(text_chunk_l[inter3])

    print("{} Retrieved Chunks:".format(show_time()), neib)

    return list(neib), retrieve_text


if __name__ == '__main__':
    pass