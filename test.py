from FlagEmbedding import BGEM3FlagModel
import json


model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
action_reason_file = json.load(open('../Data/PittAd/train/QA_Combined_Action_Reason_train.json'))
alignment_file = 'IN_InternVL_AR_SDXL_20240613_204248_description_single_paragraph_no_textQWenLM_text_image_alignment_isFineTunedTrue_3000_weighted.json'
alignment = json.load(open(f'../experiments/results/{alignment_file}'))
alignment_score = {}

for image_url in alignment:
    if alignment[image_url] == [0, 0, 0, 0]:
        alignment_score[image_url] = [0, 0, 0, 0]
        with open(f'../experiments/results/new_{alignment_file}', "w") as outfile:
            json.dump(alignment_score, outfile)
        continue
    similarity_score = 0
    similarity_scores_action = []
    similarity_scores_reason = []
    action_reasons = action_reason_file[image_url][0]
    generated_image_message = alignment[image_url][0].lower()
    for action_reason in action_reasons:
        print(action_reason)
        action_reason = action_reason.lower()

        similarity_score_action = model.compute_score([action_reason.split('because')[0],
                                                       generated_image_message.split('because')[0]],
                                                       max_passage_length=128,
                                                       weights_for_different_modes=[0.4, 0.2, 0.4])[
            'colbert+sparse+dense']
        similarity_score_reason = model.compute_score([action_reason.split('because')[-1],
                                                            generated_image_message.split('because')[-1]],
                                                           max_passage_length=128,
                                                           weights_for_different_modes=[0.4, 0.2, 0.4])[
            'colbert+sparse+dense']
        similarity_score += (similarity_score_action + similarity_score_reason * 4) / 5
        print(similarity_score)
        similarity_scores_action.append(similarity_score_action)
        similarity_scores_reason.append(similarity_score_reason)

    similarity_score = similarity_score / len(action_reasons)
    alignment_score[image_url] = [generated_image_message,
                                   similarity_score,
                                   similarity_scores_action,
                                   similarity_scores_reason]
    with open(f'../experiments/results/new_{alignment_file}', "w") as outfile:
        json.dump(alignment_score, outfile)

score = 0
count = 0
for image_url in alignment_score:
    score += alignment_score[image_url][1]
    count += 1
print(f'average score: {score/count}')