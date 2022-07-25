import argparse


def load_arguments(dataset=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_number",
                           default=50,
                           type=int,
                           help="the number of samples")
    parser.add_argument("--sample_batch_size",
                           default=50,
                           type=int,
                           help="the batch size of sampling")
    parser.add_argument("--fill_mask_model",
                           default="BART",
                           type=str,
                           help="model for phrase mask filling")
    parser.add_argument("--sentiment_batch_size",
                           default=64,
                           type=int,
                           help="the batch size of sentiment scoring")
    parser.add_argument("--prob_batch_size",
                           default=512,
                           type=int,
                           help="the batch size for original prob calculation")
    parser.add_argument("--dataset",
                           default="yelp",
                           type=str,
                           help="the dataset to be attacked")
    parser.add_argument("--ll_ratio_threshold",
                           default=1,
                           type=float,
                           help="the threshold of likelihood ratio")
    parser.add_argument("--topk",
                           default=50,
                           type=int,
                           help="the topk number during sampling")
    parser.add_argument("--attack_unit",
                           default="phrases",
                           type =str,
                           help="which unit type to attack, default is phrase")
    parser.add_argument("--ll_model",
                           default="roberta",
                           type=str,
                           help="The model used for likelihood estimation")
    parser.add_argument("--use_use",
                           default="True",
                           type=str,
                           help="whether to use USE similarity filtering or not")
    parser.add_argument("--use_ll",
                           default="False",
                           type=str,
                           help="whether to use likelihood ratio filtering or not")
    parser.add_argument("--use_threshold",
                           default=0.7,
                           type=float,
                           help="the filtering threshold for USE similarity")
    parser.add_argument("--use_window",
                           default=30,
                           type=int,
                           help="window size when using USE similarity for filtering")
    parser.add_argument("--len_filter",
                           default="True",
                           type=str,
                           help="whether to use length filtering or not")
    parser.add_argument("--phrase_depth",
                           default=4,
                           type=int,
                           help="the maximum depth of extracted phrases")
    parser.add_argument("--length_restriction",
                           default=3,
                           type=int,
                           help="the length restriction for filtering generated phrases candidates")

    args = parser.parse_args()
    print(args)
    return args