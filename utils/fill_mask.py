import os
import pickle
import re
import torch
import copy
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import numpy as np


def text_infilling(local_sentences, word_list, phrase_info, model, label, ll_model, ll_tokenizer,
                   sample_number, sample_batch_size, fill_mask_model, sentiment_batch_size, ll_ratio_threshold, topk,
                   use_ll, use_use, use_threshold, use_window, len_filter, length_restriction, orig_example, logger):
    # basic information
    phrase = phrase_info[0]
    phrase_sent = phrase_info[2]
    sents_copy = copy.deepcopy(local_sentences)
    start_index = phrase_info[3][0]
    end_index = phrase_info[3][1]
    words = word_list[phrase_sent]

    # begin blank filling
    # mask the selected phrases
    tmp_words = words[:start_index] + ["<mask>"] + words[end_index+1:]
    sents_copy[phrase_sent] = " ".join(tmp_words)
    context = " ".join(sents_copy)
    context = process_string(context)

    # get the index of the mask token
    span_start_index = context.index("<mask>")
    span_end_index = context.index("<mask>") + 5 - len(context)

    # get the string before and after mask
    before_string = context[:span_start_index]
    after_string = context[span_end_index + 1:]

    # prepare span list
    gen_span_list = []

    # calculate the sample frequency
    sample_frequency = sample_number // sample_batch_size
    n_left = sample_number % sample_batch_size

    # start to infill
    with torch.no_grad():
        for i in range(sample_frequency):
            fillings = model.fill_mask([context], topk=sample_batch_size, sampling=True, sampling_topk=topk,
                                      temperature=1., match_source_len=False)
            torch.cuda.empty_cache()
            for g in fillings[0]:
                if g[0][:span_start_index] == before_string:
                    if g[0][span_end_index + 1:] == after_string:
                        gen_span_list.append(g[0][span_start_index: span_end_index + 1].strip())

        if n_left > 0:
            fillings = model.fill_mask([context], topk=n_left, sampling=True, sampling_topk=topk,
                                      temperature=1., match_source_len=False)
            torch.cuda.empty_cache()
            for g in fillings[0]:
                if g[0][:span_start_index] == before_string:
                    if g[0][span_end_index + 1:] == after_string:
                        gen_span_list.append(g[0][span_start_index: span_end_index + 1].strip())
        logger.info("number of total samples: %d" % (sample_number))
        logger.info("nunmber of examples that keep the original sentences: %d" % len(gen_span_list))

    # remove duplicated phrases
    gen_span_set = set(gen_span_list)
    gen_span_list = list(gen_span_set)
    logger.info("nunmber of total distinct examples: %d" % len(gen_span_list))

    # length filter
    _infill_span = []   # list storing spans after length filetering
    if len_filter == "True":
        # filter the span by span length and filter out span with punctutation
        for span_text in gen_span_list:
            # default length restriction 3
            if len(span_text.split()) <= len(phrase.split()) + length_restriction:
                if "." not in span_text:
                    if "," not in span_text:
                        if ";" not in span_text:
                            if len(span_text) != 0:
                                _infill_span.append(span_text)
    else:
        _infill_span = gen_span_list
    logger.info("number of examples meeting length and punctutation threshold: %d" % len(_infill_span))

    # if no phrases is left, return none
    if len(_infill_span) == 0:
        return [], []

    #  filter by sentiment score
    logger.info("begin computing likelihoods")
    sentiment_batch_size = sentiment_batch_size
    pos_sent_ids_list = []
    neg_sent_ids_list = []
    sent_index_list = []
    replaced_id_list = []
    token_index_list = []

    # prepare the batch input
    for span_i, span in enumerate(_infill_span):
        span_text = span
        # reconstruct span_text to avoid decoding or type error
        span_text = " ".join(span_text.split())
        new_word_list = words[:start_index] + span_text.split() + words[end_index + 1:]
        new_sent = " ".join(new_word_list)

        if start_index == 0:
            span_text = span_text
        else:
            span_text = " " + span_text

        span_ids = ll_tokenizer.convert_tokens_to_ids(ll_tokenizer.tokenize(span_text))
        len_span_ids = len(span_ids)
        span_start_index = len(ll_tokenizer.tokenize("<pos>" + " ".join(words[:start_index]))) + 1
        len_sentence = len(pos_sent_ids_list)

        # the function to generate masked sentence ids for each label given different phrases
        def get_masked_sents_for_one_label(new_sent, label, masked_sent_ids_list, get_auxiliary_indices=False):
            labeled_sent = label + new_sent
            labeled_sent_ids = ll_tokenizer.encode(labeled_sent)
            tmp_masked_sents_ids = []
            tmp_sent_index_list, tmp_token_index_list, tmp_replaced_id_list = [], [], []
            for token_i in range(len_span_ids):
                tmp_ids = labeled_sent_ids[:]
                replaced_id = tmp_ids[span_start_index + token_i]
                tmp_ids[span_start_index + token_i] = ll_tokenizer.mask_token_id
                tmp_masked_sents_ids.append(tmp_ids)
                if get_auxiliary_indices:
                    sent_index = len_sentence + token_i
                    token_index = span_start_index + token_i
                    tmp_replaced_id_list.append(replaced_id)
                    tmp_sent_index_list.append(sent_index)
                    tmp_token_index_list.append(token_index)
            masked_sent_ids_list.extend(tmp_masked_sents_ids)
            return masked_sent_ids_list, tmp_replaced_id_list, tmp_sent_index_list, tmp_token_index_list

        # get masked sentences ids by adding each special token
        pos_sent_ids_list, tmp_replaced_id_list, tmp_sent_index_list, tmp_token_index_list = \
            get_masked_sents_for_one_label(new_sent, "<pos>", pos_sent_ids_list, get_auxiliary_indices=True)
        neg_sent_ids_list, _, _, _ = get_masked_sents_for_one_label(new_sent, "<neg>", neg_sent_ids_list)
        sent_index_list.append(tmp_sent_index_list)
        replaced_id_list.extend(tmp_replaced_id_list)
        token_index_list.extend(tmp_token_index_list)

    # get likelihood for masked sentence in different labels
    def get_masked_likelihood(masked_sent_ids_list):
        max_len = max(len(sent) for sent in masked_sent_ids_list)
        input_ids = torch.tensor([sent + [ll_tokenizer.pad_token_id] * (max_len - len(sent))
                     for sent in masked_sent_ids_list], dtype=torch.long)
        attention_mask = torch.tensor(
            [[1] * len(sent) + [0] * (max_len - len(sent)) for sent in masked_sent_ids_list], dtype=torch.long)
        dataset = TensorDataset(input_ids, attention_mask)
        sampler = SequentialSampler(dataset)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=sentiment_batch_size)
        lls = []
        with torch.no_grad():
            ll_model.eval()
            offset = 0
            for input_ids, input_mask in data_loader:
                input_ids = input_ids.to("cuda")
                input_mask = input_mask.to("cuda")
                outputs = ll_model(input_ids=input_ids, attention_mask=input_mask)[0].detach()
                cur_pos_index = torch.tensor(token_index_list[offset: offset + outputs.size(0)], device="cuda")\
                    .unsqueeze(-1).unsqueeze(-1).repeat(1, 1, outputs.size(-1))
                mask_ll = torch.gather(outputs, 1, cur_pos_index).softmax(dim=-1).squeeze(1).cpu()
                cur_token_index = torch.tensor(replaced_id_list[offset: offset + outputs.size(0)]).unsqueeze(-1)
                cur_lls = torch.gather(mask_ll, 1, cur_token_index).squeeze(-1)
                lls.append(cur_lls)
                offset += outputs.size(0)
        return torch.cat(lls)

    neg_lls = get_masked_likelihood(neg_sent_ids_list)
    pos_lls = get_masked_likelihood(pos_sent_ids_list)
    all_lls = [neg_lls, pos_lls]

    # get the sentiment score
    sentiment_list = []
    if label == 0:
        gold_lls = all_lls.pop(0)
    if label == 1:
        gold_lls = all_lls.pop(1)
    other_lls = all_lls

    for span_index in range(len(_infill_span)):
        gold_ll = 1
        cur_other_lls = torch.tensor([1], dtype=torch.float)

        for sent_index in sent_index_list[span_index]:
            gold_ll *= gold_lls[sent_index]
            cur_other_lls *= torch.cat([other_lls[tmp_i][sent_index].unsqueeze(0) for tmp_i in range(1)])

        final_other_ll = torch.max(cur_other_lls)

        sentiment_list.append((gold_ll/final_other_ll))

    sentiment_list = np.array(sentiment_list)
    sentiment_mask = sentiment_list > ll_ratio_threshold
    sentiment_index = [index for index, value in enumerate(sentiment_mask) if value == True]


    infill_text = []
    infill_span = []
    for index_i in sentiment_index:
        span_text = _infill_span[index_i].strip()
        infill_span.append(span_text)
        new_sentences = copy.deepcopy(local_sentences)
        new_sentences[phrase_info[2]] = " ".join(words[:start_index] + span_text.split() + words[end_index + 1:])
        infill_text.append(new_sentences)
    logger.info("number of examples meeting sentiment threshold: %d" % len(infill_span))

    return infill_text, infill_span


def process_string(string):
    string = re.sub("( )(\'[(m)(d)(t)(ll)(re)(ve)(s)])", r"\2", string)
    string = re.sub("(\d+)( )([,\.])( )(\d+)", r"\1\3\5", string)
    # U . S . -> U.S.
    string = re.sub("(\w)( )(\.)( )(\w)( )(\.)", r"\1\3\5\7", string)
    # reduce left space
    string = re.sub("( )([,\.!?:;)])", r"\2", string)
    # reduce right space
    string = re.sub("([(])( )", r"\1", string)
    string = re.sub("s '", "s'", string)
    # reduce both space
    string = re.sub("(')( )(\S+)( )(')", r"\1\3\5", string)
    string = re.sub("(\")( )(\S+)( )(\")", r"\1\3\5", string)
    string = re.sub("(\w+) (-+) (\w+)", r"\1\2\3", string)
    string = re.sub("(\w+) (/+) (\w+)", r"\1\2\3", string)
    # string = re.sub(" ' ", "'", string)
    return string






