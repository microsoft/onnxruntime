import argparse
import csv
import math
import sys

# Calculate the Prediction Gain for a given candidate, ideal reference word at a given rank
sys.stdout.reconfigure(encoding='utf-8')


def prediction_gain(rem_pref, candidate, rem_query, current_rank):
    gain = 0
    if (len(rem_query) >= len(candidate)):
        cand_list = list(candidate)
        rem_pref_list = list(rem_pref)
        rem_pref_len = len(rem_pref_list)
        cand_len = len(cand_list)
        rem_query_list = list(rem_query)
        overlap = 0
        i = 0
        while(i < len(rem_query_list) and i < len(cand_list)):
            if (rem_query_list[i] == cand_list[i]):
                overlap += 1
            else:
                break
            i += 1
        # Only reward the candidates on word boundaries
        if (((i < len(rem_query_list) and rem_query_list[i] == ' ') or (i == len(rem_query_list))) and overlap > 0 and overlap == cand_len):
            ks = cand_len - rem_pref_len
            gain = ks/math.log2(1+current_rank)
    return gain

# Calculate the Prefix Penalty


def prefix_penalty(prefixlen):
    penalty = 1
    if (prefixlen <= 12):
        penalty = math.log(prefixlen+1, 12)
    return penalty

# Calculate the APPG Scores for a given Ref Prediction and Model Prediction file pair


def calculate_appg(refPredictions, modPredictions, verbFlag, top=None):
    ref_dict = {}
    mod_dict = {}
    ref_pref_dict = {}
    num_unique_ref_prefixes = 0
    num_pq_pairs = 0
    with open(refPredictions, encoding='utf-8') as ref:
        rd = csv.reader(ref, delimiter="\t")
        for row in rd:
            unique_key = row[0] + "____||||____||||____" + row[1]
            if (unique_key in ref_dict):
                print("Duplicate P, Q Pair Found in Ref Predictions: %s, %s" % (row[0], row[1]))
                return
            else:
                num_pq_pairs += 1
                ref_dict[unique_key] = 1
                if (row[0] not in ref_pref_dict):
                    ref_pref_dict[row[0]] = 1
                    num_unique_ref_prefixes += 1

    cov_prefixes = 0
    total_mod_suggs = 0
    with open(modPredictions, 'r', encoding='utf-8') as mod:
        rd = csv.reader(mod, delimiter="\t")
        for row in rd:
            pref = row[0]
            rank_list = row[1].rstrip(',')
            if (rank_list != "" and rank_list != "NULL"):
                if top and top > 0:
                    rank_list = ','.join(rank_list.split(',')[0:top])

                if (pref not in mod_dict):
                    mod_dict[pref] = rank_list
                else:
                    print("Duplicate Prefix Found in Model Predictions: %s" % (pref))
                    return
                # Maintain the number of suggestions per prefix and num of prefixes with non-NULL predictions
                curr_sugg_num = len(rank_list.split(','))
                total_mod_suggs += curr_sugg_num
                #print("%s\t%d" %(row[0],curr_sugg_num))
                cov_prefixes += 1

    total_ppg_score = 0
    total_best_gain = 0
    non_zero_gain_pairs = 0
    nonzero_gain_pref = {}
    nonzero_gain_pref_cov = 0
    model_predictions = 0
    for unique_key in ref_dict.keys():
        pref = unique_key.split("____||||____||||____")[0]
        query = unique_key.split("____||||____||||____")[1]

        space_index = pref.rfind(" ")
        if (space_index == -1):
            remaining_query = query
            remaining_pref = pref
        else:
            remaining_query = query[space_index+1:]
            remaining_pref = pref[space_index+1:]

        ppg_score = 0
        if (pref in mod_dict):
            model_predictions += 1
            model_predictions_list = mod_dict.get(pref)
            rank_list = model_predictions_list.split(',')
            rank = 1
            max_gain = 0
            for cand in rank_list:
                gain = prediction_gain(remaining_pref, cand, remaining_query, rank)
                if (gain > max_gain):
                    max_gain = gain
                rank += 1

            penalty = prefix_penalty(len(pref))
            ppg_score = max_gain * penalty
            total_ppg_score += ppg_score
            if (max_gain > 0):
                non_zero_gain_pairs += 1
                if (pref not in nonzero_gain_pref):
                    nonzero_gain_pref[pref] = 1
                    nonzero_gain_pref_cov += 1

            if (verbFlag):
                print("%s\t%s\t%s\t%f" % (pref, query, rank_list, ppg_score))
        else:
            if (verbFlag):
                print("%s\t%s\t%s\t%f" % (pref, query, "NULL", 0))

        # Calculate the maximum possible prediction gain
        max_prediction_gain = (len(query)-len(pref)) * prefix_penalty(len(pref))
        total_best_gain += max_prediction_gain

    appg_score = 0
    max_appg_score = 0
    pref_cov = 0
    avg_suggs = 0
    if (num_unique_ref_prefixes > 0):
        pref_cov = cov_prefixes/num_unique_ref_prefixes
    if (cov_prefixes > 0):
        avg_suggs = total_mod_suggs/cov_prefixes
    if (num_pq_pairs > 0):
        appg_score = total_ppg_score/num_pq_pairs
    if (num_pq_pairs > 0):
        max_appg_score = total_best_gain/num_pq_pairs

    print("Number of Unique Prefixes: %d" % num_unique_ref_prefixes)
    print("Num P,Q Pairs: %d" % num_pq_pairs)
    print("Num of P, Q Pairs with Non-NULL Model Predictions:  %d" % model_predictions)
    print("Num of P, Q Pairs with Non-Zero Gain: %d" % non_zero_gain_pairs)
    print("Num of Prefixes with Non-Zero Gain:  %d" % nonzero_gain_pref_cov)
    print("Model Triggering Coverage (Prefix Level): %f" % pref_cov)
    print("Avg. No. of Suggs Per Prefix: %f" % avg_suggs)
    print("Model APPG Score: %f" % appg_score)
    print("Max APPG Score: %f" % max_appg_score)
    print("Total Suggestions: %f" % total_mod_suggs)


def main():
    parser = argparse.ArgumentParser(description='Evaluation Script for Next Word Prediction Models')
    parser.add_argument('--refFile',
                        metavar='referenceFile',
                        type=str,
                        help='File Containing Reference Pre, Query pairs (Tab Separated)')

    parser.add_argument('--modelFile',
                        metavar='modelFile',
                        type=str,
                        help='File Containing Model Outputs Pre, NW List (Tab Separated)')

    parser.add_argument('--top',
                        metavar='top',
                        type=int,
                        default=None,
                        help='Top K canditates to calculate APPG Scores')

    parser.add_argument('--verbose', help='Print P,Q Level APPG Scores',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    refFile = args.refFile
    modFile = args.modelFile
    verboseFlag = args.verbose
    calculate_appg(refFile, modFile, verboseFlag, args.top)


if __name__ == "__main__":
    main()
