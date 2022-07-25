import copy
import logging
import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import RobertaForMaskedLM, RobertaTokenizer
# sys.path.append(os.path.dirname(sys.path[0]))
from utils.bert_classifier import BERTinfer
from utils.BERT_cls_loader import BERTClsDataloader
from utils.config import load_arguments
from utils.dataloader import read_corpus
from utils.extract_phrase import examples2phrases
from utils.fill_mask import text_infilling
from utils.phrase_importance_rank import get_attack_order


def get_logger(log_path):
    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        level=logging.INFO)
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    return logger

if __name__ == "__main__":
    # load arguments
    args = load_arguments()
    dataset = args.dataset
    accelerator = Accelerator()

    # create the log and adversarial example txt file
    logger = get_logger("log_%s_%d_%d_%d" %
                        (args.dataset, args.sample_number, args.phrase_depth, args.length_restriction))
    attack_txt = open("result_%s_%d_%d_%d" %
                      (args.dataset, args.sample_number, args.phrase_depth, args.length_restriction), 'w')
    logger.info(args)

    # load data
    examples = read_corpus("./data/%s.tsv" % dataset)

    # delete after organizing
    examples = examples[:5]

    # load parsed phrases, sentences and word lists
    all_phrases, all_sentences, all_word_lists = examples2phrases(examples, "./stanford-corenlp-4.2.2", dataset)

    # prepare target classifier
    label_map = {"yelp": ["Negatvie", "Postive"],
                   "ag": ["World", "Sport", "Bussiness", "Sci-tech"]}
    label_list = label_map[dataset]
    bert_dataset = BERTClsDataloader("uncased")
    classifier_model = BERTinfer("bert", "./classifier/%s" % args.dataset, len(label_list), "uncased",
                      bert_dataset,
                      batch_size=args.prob_batch_size,
                      accelerator=accelerator)

    # prepare mask filling model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BART = torch.hub.load('./fairseq', 'bart.base', source="local")
    BART.eval()
    BART.cuda()

    # prepare models for likelihood estimation
    ll_tokenizer = RobertaTokenizer.from_pretrained(("./likelihood_model/%s" % args.dataset))
    ll_model = RobertaForMaskedLM.from_pretrained(("./likelihood_model/%s" % args.dataset)).to("cuda").eval()

    logger.info("model loading finished!")

    # some numerical values recoding attack performance
    # number of examples that are predicted originally correct
    n = 0
    # number of examples that are attacked successfully
    success = 0
    # calculate classification probabilities for all original examples
    all_original_probs = classifier_model.get_probs_in_batch([e[1] for e in examples],
                                                                [e[0] for e in examples])
    # clean the gpu cache
    torch.cuda.empty_cache()


    logger.info("begin attack!")
    for i, example in enumerate(tqdm(examples)):
        # Input sequence
        text = example[1]
        label = example[0]
        attack_txt.write("Index:  %d\n" % i)

        # write the baisic information of the example into log
        logger.info("orginal sentence: %s", text)
        logger.info("label: %s", label_list[label])

        # get the original prob of the example
        orig_probs = all_original_probs[i]
        orig_label = torch.argmax(orig_probs).item()
        orig_prob = orig_probs.max()

        # if target model predicts incorrectly, then skip
        if orig_label != label:
            logger.info("original example can not be predicted correctly by victim model.\n")
            attack_txt.write("original example can not be predicted correctly by victim model.")
            attack_txt.write("\n"+ "_"*30 + "\n")
            continue

        # write the prediction information of the example into log
        logger.info("original probability: %.4f", orig_prob)
        attack_txt.write("original sentence: %s\n" % text)
        attack_txt.write("label: %s, original probability: %.4f\n" % (label_list[label], orig_prob))

        # increase number of total examples predicted correctly by 1
        n = n + 1

        # get phrases, local sentences and word list produced by parsing tree
        phrases = all_phrases[i]
        local_sentences = all_sentences[i]
        word_list = all_word_lists[i]

        # copy the sentences for future sentence reconstruction
        local_sents_copy = copy.deepcopy(local_sentences)

        # get attack order
        attack_order = get_attack_order(local_sentences, label, phrases, word_list, classifier_model)
        logger.info("attack order:  %s", "    ".join(e[0] for e in attack_order))
        attack_txt.write("attack order:  %s\n" % "    ".join(e[0] for e in attack_order))

        #intial best sentences
        best_sentences = local_sents_copy

        len_attack = len(attack_order)

        # begin attack
        for attack_i in range(len_attack):
            phrase_info = attack_order[attack_i]
            phrase = phrase_info[0]
            type = phrase_info[1]

            gen, gen_spans = text_infilling(best_sentences, word_list, phrase_info, BART, label, ll_model,
                                            ll_tokenizer, args.sample_number, args.sample_batch_size,
                                            args.fill_mask_model, args.sentiment_batch_size, args.ll_ratio_threshold, args.topk, args.use_ll,
                                            args.use_use, args.use_threshold, args.use_window, args.len_filter, args.length_restriction, text, logger)

            if len(gen) == 0:
                logger.info("no candadites of span %s meets the filtering requirements" % phrase )
                attack_txt.write("no candadites of span %s meets the filtering requirements\n" % phrase )
                continue

            torch.cuda.empty_cache()

            # search for the strongest attack phrase
            gen_list = [" ".join(e) for e in gen]
            label_list = [label] * len(gen_list)
            attack_prob = classifier_model.get_prob_in_batch(gen_list, label_list)
            attack_probs = classifier_model.get_probs_in_batch(gen_list, label_list)
            torch.cuda.empty_cache()
            best_prob = attack_prob.min()
            if best_prob > orig_prob:
                best_span = phrase
                best_sentences = best_sentences
                best_prob = orig_prob
                best_label = label
            else:
                best_index = attack_prob.argmin()
                best_sentences = gen[best_index]
                best_span = gen_spans[best_index]
                best_label = torch.argmax(attack_probs[best_index]).item()
            logger.info("attacked phrase--> %s, new phrase--> %s, orig_prob--> %.4f, attack_prob--> %.4f",
                        phrase, best_span, orig_prob, best_prob)
            attack_txt.write("attacked phrase--> %s, new phrase--> %s, orig_prob--> %.4f, attack_prob--> %.4f\n" %
                             (phrase, best_span, orig_prob, best_prob))
            orig_prob = best_prob
            torch.cuda.empty_cache()

            # change the word list and attack order index
            sent_index = phrase_info[2]
            phrase_start_index = phrase_info[3][0]
            phrase_end_index = phrase_info[3][1]
            span_list = best_span.split()
            len_span = len(span_list)
            del word_list[sent_index][phrase_start_index:phrase_end_index+1]
            for indexx in range(len_span):
                word_list[sent_index].insert(phrase_start_index+indexx, span_list[indexx])
            for attackk in attack_order:
                if attackk[2] == sent_index:
                    if attackk[3][0] > phrase_start_index:
                        attackk[3][0] += len_span - (phrase_end_index-phrase_start_index+1)
                        attackk[3][1] += len_span - (phrase_end_index-phrase_start_index+1)

            if best_label != label:
                success = success + 1
                logger.info("attack success!")
                logger.info("adv sentence--> %s", " ".join(best_sentences))
                logger.info("adv label--> %s", label_list[best_label])
                attack_txt.write("adv sentence--> %s\n" % " ".join(best_sentences))
                attack_txt.write("adv label--> %s\n" % label_list[best_label])
                break

        if best_label == label:
            logger.info("attack fail!")
            logger.info("failed adv sentence--> %s", " ".join(best_sentences))
            attack_txt.write("failed adv sentence--> %s\n" % " ".join(best_sentences))
        total_example = i + 1
        accuracy = n / total_example
        success_rate = success / n
        logger.info(
            "total examples--> %d, original_accuracy--> %.4f, success_rate--> %.4f\n\n",
            total_example, accuracy, success_rate)
        attack_txt.write(
            "total examples--> %d, original_accuracy--> %.4f, success_rate--> %.4f\n" %
             (total_example, accuracy, success_rate))
        attack_txt.write("_"*30+"\n\n")

