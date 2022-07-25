import copy


# Using the model to predict the importance of each phrase
def get_attack_order(local_sentences, label, phrases, word_list, classifier_model):
    # mask or delete the phrases and compute probability drops
    def get_masked_sentences(local_sentences):
        masked_sentences, masked_phrase_indices, targeted_phrases = [], [], []
        for i, phrase in enumerate(phrases):
            phrase_text = phrase[0]
            sent_index = phrase[2]
            start_index = phrase[-1][0]
            end_index = phrase[-1][1]

            # copy the original sentence, word list and replace the selected phrase with mask tokens
            tmp_sentences = copy.deepcopy(local_sentences)
            tmp_word_list = copy.deepcopy(word_list)
            words = tmp_word_list[sent_index]
            for i in range(end_index-start_index+1):
                words[start_index+i] =  "<mask>"
            tmp_sentences[sent_index] = " ".join(words)

            masked_sentences.append(" ".join(tmp_sentences))
            masked_phrase_indices.append([start_index, end_index])
            targeted_phrases.append(phrase)
        return masked_sentences, masked_phrase_indices, targeted_phrases

    # Get the list of sentences where each phrase is masked
    masked_sentences, masked_phrase_indices, phrases = get_masked_sentences(local_sentences)
    tmp_labels = [label] * len(masked_sentences)
    all_tmp_probs = classifier_model.get_prob_in_batch(masked_sentences, tmp_labels)

    # sort the phrases according to their corresponding prediction probabilities
    for i in range(len(phrases)):
        phrases[i].append((all_tmp_probs[i]).item())
    phrases = sorted(phrases, key=lambda x: x[4])

    attack_order = []
    replaced_words = {}
    # delete the overlapped phrases
    tmp_word_list = copy.deepcopy(word_list)
    for phrase in phrases:
        sent_index = phrase[2]
        start_index = phrase[3][0]
        end_index = phrase[3][1]
        words = tmp_word_list[sent_index]
        for i in range(start_index, end_index+1):
            if words[i] != "<replace>":
                words[i] = "<replace>"
                if i == end_index:
                    attack_order.append(phrase)
            else:
                break


    return attack_order



