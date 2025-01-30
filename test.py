# from FlagEmbedding import BGEM3FlagModel
import json
#
#
# model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
# action_reason_file = json.load(open('../Data/PittAd/train/QA_Combined_Action_Reason_train.json'))
# alignment_file = 'IN_InternVL_AR_SDXL_20240613_204248_description_single_paragraph_no_textQWenLM_text_image_alignment_isFineTunedTrue_3000_weighted.json'
# alignment = json.load(open(f'../experiments/results/{alignment_file}'))
# alignment_score = {}
#
# for image_url in alignment:
#     if alignment[image_url] == [0, 0, 0, 0]:
#         alignment_score[image_url] = [0, 0, 0, 0]
#         with open(f'../experiments/results/new_{alignment_file}', "w") as outfile:
#             json.dump(alignment_score, outfile)
#         continue
#     similarity_score = 0
#     similarity_scores_action = []
#     similarity_scores_reason = []
#     action_reasons = action_reason_file[image_url][0]
#     generated_image_message = alignment[image_url][0].lower()
#     for action_reason in action_reasons:
#         print(action_reason)
#         action_reason = action_reason.lower()
#
#         similarity_score_action = model.compute_score([action_reason.split('because')[0],
#                                                        generated_image_message.split('because')[0]],
#                                                        max_passage_length=128,
#                                                        weights_for_different_modes=[0.4, 0.2, 0.4])[
#             'colbert+sparse+dense']
#         similarity_score_reason = model.compute_score([action_reason.split('because')[-1],
#                                                             generated_image_message.split('because')[-1]],
#                                                            max_passage_length=128,
#                                                            weights_for_different_modes=[0.4, 0.2, 0.4])[
#             'colbert+sparse+dense']
#         similarity_score += (similarity_score_action + similarity_score_reason * 4) / 5
#         print(similarity_score)
#         similarity_scores_action.append(similarity_score_action)
#         similarity_scores_reason.append(similarity_score_reason)
#
#     similarity_score = similarity_score / len(action_reasons)
#     alignment_score[image_url] = [generated_image_message,
#                                    similarity_score,
#                                    similarity_scores_action,
#                                    similarity_scores_reason]
#     with open(f'../experiments/results/new_{alignment_file}', "w") as outfile:
#         json.dump(alignment_score, outfile)
#
# score = 0
# count = 0
# for image_url in alignment_score:
#     score += alignment_score[image_url][1]
#     count += 1
# print(f'average score: {score/count}')
file_p = 'IN_InternVL_LLM_input_LLAMA3_FTFalse_DALLE3_20250128_234330_description_single_paragraph_full_description_persuasion_creativity.json'
p_star_file = json.load(open(f'../experiments/results/{file_p}'))
file_a = 'new_IN_InternVL_AR_DALLE3_20250128_234450_description_single_paragraph_full_descriptionLLAMA3_instruct_text_image_alignment_isFineTunedTrue_3000_weighted.json'
alignment_file = json.load(open(f'../experiments/results/{file_a}'))

PA_scores = {}

for image_url in alignment_file:
    if image_url not in p_star:
        continue
    p_star = p_star_file[image_url]
    alignment_score = sum(alignment_file[image_url][3])/len(alignment_file[image_url][3])
    score = 0
    count = 0
    for i, value in enumerate(p_star):
        if i not in [3, 6]:
            score += value/5
            count += 1
        score += alignment_score
        count += 1
    PA_scores[image_url] = score/count
    with open(f'../experiments/results/new_{file_p}', "w") as outfile:
        json.dump(PA_scores, outfile)
score = 0
count = 0
for image_url in PA_scores:
    score+= PA_scores[image_url]
    count+=1
print(score/count)