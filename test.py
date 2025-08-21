# from FlagEmbedding import BGEM3FlagModel
# import json


# model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
# action_reason_file = json.load(open('../Data/PittAd/train/QA_Combined_Action_Reason_train.json'))
# # alignment_file = 'LLM_input_QWenLM_FTFalse_PSALLAMA3_instruct_text_image_alignment_isFineTunedTrue_3000_weighted.json_Flux_20250225_003919LLAMA3_instruct_text_image_alignment_isFineTunedTrue_3000_weighted.json'
# while True:
#     alignment_file = input('input file:')
#     print('input recieved:', alignment_file)
#     alignment = json.load(open(f'../experiments/results/{alignment_file}'))
#     alignment_score = {}
#     for image_url in alignment:
#         if image_url == 'average':
#             continue
#         if alignment[image_url] == [0, 0, 0, 0]:
#             alignment_score[image_url] = [0, 0, 0, 0]
#             with open(f'../experiments/results/{alignment_file}', "w") as outfile:
#                 json.dump(alignment_score, outfile)
#             continue
#         similarity_score = 0
#         similarity_scores_action = []
#         similarity_scores_reason = []
#         action_reasons = action_reason_file[image_url][0]
#         generated_image_message = alignment[image_url][0].lower()
#         for action_reason in action_reasons:
#             print(action_reason)
#             action_reason = action_reason.lower()

#             similarity_score_action = model.compute_score([action_reason.split('because')[0],
#                                                            generated_image_message.split('because')[0]],
#                                                            max_passage_length=128,
#                                                            weights_for_different_modes=[0.4, 0.2, 0.4])[
#                 'colbert+sparse+dense']
#             similarity_score_reason = model.compute_score([action_reason.split('because')[-1],
#                                                                 generated_image_message.split('because')[-1]],
#                                                                max_passage_length=128,
#                                                                weights_for_different_modes=[0.4, 0.2, 0.4])[
#                 'colbert+sparse+dense']
#             similarity_score += (similarity_score_action + similarity_score_reason * 4) / 5
#             print(similarity_score)
#             similarity_scores_action.append(similarity_score_action)
#             similarity_scores_reason.append(similarity_score_reason)

#         similarity_score = similarity_score / len(action_reasons)
#         alignment_score[image_url] = [generated_image_message,
#                                        similarity_score,
#                                        similarity_scores_action,
#                                        similarity_scores_reason]
#         with open(f'../experiments/results/{alignment_file}', "w") as outfile:
#             json.dump(alignment_score, outfile)
#             print('writing done')

#     score = 0
#     count = 0
#     for image_url in alignment_score:
#         score += alignment_score[image_url][1]
#         count += 1
#     print(f'average score: {score/count}')


# #
# # file_p = 'IN_InternVL_LLM_input_QWenLM_FTFalse_AuraFlow_20250127_124801_description_single_paragraph_full_description_persuasion_creativity.json'
# # p_star_file = json.load(open(f'../experiments/results/{file_p}'))
# # file_a = 'new_IN_InternVL_LLM_input_QWenLM_FTFalse_AuraFlow_20250127_124801_description_single_paragraph_full_descriptionVILA_text_image_alignment_isFineTunedTrue_3000_weighted.json'
# # alignment_file = json.load(open(f'../experiments/results/{file_a}'))
# #
# # PA_scores = {}
# #
# # for image_url in alignment_file:
# #     if image_url not in p_star_file:
# #         continue
# #     if alignment_file[image_url][0] == 0:
# #         PA_scores[image_url] = 0
# #         with open(f'../experiments/results/new_{file_p}', "w") as outfile:
# #             json.dump(PA_scores, outfile)
# #         continue
# #     p_star = p_star_file[image_url]
# #     alignment_score = alignment_file[image_url][1]
# #     score = 0
# #     count = 0
# #     for i, value in enumerate(p_star):
# #         if i not in [3, 6]:
# #             score += value/5
# #             count += 1
# #         score += alignment_score
# #         count += 1
# #     PA_scores[image_url] = score/count
# #     with open(f'../experiments/results/new_{file_p}', "w") as outfile:
# #         json.dump(PA_scores, outfile)
# # score = 0
# # count = 0
# # for image_url in PA_scores:
# #     score+= PA_scores[image_url]
# #     count+=1
# # print(score/count)

# #
# # from LLMs.LLM import LLM
# # from configs.evaluation_config import get_args
# #
# # args = get_args()
# # args.LLM = 'QWenLM'
# # description = '''
# # The image features an animated scene with a mouse and a cat. The mouse, with large ears and wide eyes, is leaping through the air, appearing surprised or alarmed. Its fur is light gray, and its tail and feet are a reddish color. In the background, a small kitten with gray fur and pink inner ears is running on the ground, looking up at the mouse. The setting seems to be an outdoor path with a blurred background, suggesting motion and a dynamic interaction between the two characters. The lighting is soft and natural, enhancing the whimsical and lively atmosphere of the scene.
# # '''
# # format = '''The interpretation format is: ${object1} is ${action} ${object2}. Where object1 and object2 are either cat or mouse. ONLY RETURN A SINGLE SENTENCE IN THIS FORMAT'''
# # prompt = f"""What is the correct interpretation for the described image:
# #
# #              Description: {description}.
# #              {format}"""
# #
# # pipe = LLM(args)
# # print(pipe(prompt))


# import json

# alignment_file = json.load(open('/users/aysanaghazadeh/Downloads/IN_InternVL_AR_AuraFlow_20240816_214421_description_single_paragraph_full_descriptionVILA_text_image_alignment_isFineTunedTrue_3000_weighted.json'))
# creativity_file = json.load(open('/users/aysanaghazadeh/Downloads/AR_AuraFlow_20240816_214421weighted_creativity.json'))

# clip_sim_sum = 0
# count = 0
# for image_url in creativity_file:
#     if image_url not in alignment_file:
#         continue
#     creativity_score = creativity_file[image_url]
#     alignment_score = alignment_file[image_url][1]
#     print(f'creativity score: {creativity_score}')
#     print(f'alignment score: {alignment_score}')
#     clip_sim = alignment_score / creativity_score - 0.01
#     clip_sim_sum += clip_sim
#     count += 1
# print(f'Average clip sim is: {clip_sim_sum/count}')

# import json 


# SDXL_com = json.load(open('/users/aysanaghazadeh/Downloads/AR_SDXL_20240613_204248VILA_text_image_alignment_isFineTunedTrue_3000_weighted.json'))
# AuraFlow_com = json.load(open('/users/aysanaghazadeh/Downloads/LLAMA3Instruct_descriptions_AuraFlow_20240817_185858VILA_text_image_alignment_isFineTunedTrue_3000_weighted.json'))
# SDXL_psa = json.load(open('/users/aysanaghazadeh/Downloads/IN_InternVL_AR_SDXL_20241012_005132_description_single_paragraph_no_textVILA_text_image_alignment_isFineTunedTrue_3000_weighted.json'))
# AuraFlow_psa = json.load(open('/users/aysanaghazadeh/Downloads/IN_InternVL_LLM_input_LLAMA3_instruct_FTFalse_PSA_AuraFlow_20240925_112154_description_single_paragraph_no_textVILA_text_image_alignment_isFineTunedTrue_3000_weighted.json'))

# com = ['8/138668.jpg', '8/121068.jpg', '8/57788.jpg', '7/33807.jpg', '4/129394.jpg', '4/78954.jpg', '4/46914.jpg', '2/146302.jpg', '2/124992.jpg', '2/57982.jpg', '0/111880.jpg', '8/52488.jpg', '4/84094.jpg', '4/46914.jpg', '6/148076.jpg', '6/158126.jpg']#, '8/121068.jpg', '6/155826.jpg', '5/129255.jpg', '7/12117.jpg', '5/51325.jpg', '6/125426.jpg', '7/54357.jpg', '5/107845.jpg', '5/128995.jpg', '7/115587.jpg', '6/141276.jpg']

# PSA = ['6/143816.jpg', '8/7608.jpg', '4/102794.jpg', '4/134044.jpg', '6/17896.jpg', '7/116347.jpg', '7/46267.jpg', '6/90766.jpg', '6/17896.jpg', '6/17276.jpg', '4/102794.jpg', '3/134973.jpg', '2/116492.jpg', '0/73110.jpg', '0/45970.jpg', '0/41530.jpg']#, '0/41530.jpg', '5/71965.jpg', '0/116140.jpg', '1/57011.jpg', '3/134033.jpg', '5/71815.jpg', '1/112811.jpg', '3/81633.jpg', '4/134044.jpg', '6/143816.jpg', '7/41457.jpg']
# alphas = [1, 2, 3, 4, 5]

# print('--------------COM----------------')
# for image_url in com:
#     if AuraFlow_com[image_url][0] == 0:
#         auraflow_action_score = 0
#         auraflow_reason_score = 0
#     else:
#         auraflow_reason_score = sum(AuraFlow_com[image_url][-1])/len(AuraFlow_com[image_url][-1])
#         auraflow_action_score = sum(AuraFlow_com[image_url][-2])/len(AuraFlow_com[image_url][-2])
#     if SDXL_com[image_url][0] == 0:
#         sdxl_action_score = 0
#         sdxl_reason_score = 0
#     else:
#         sdxl_reason_score = sum(SDXL_com[image_url][-1])/len(SDXL_com[image_url][-1])
#         sdxl_action_score = sum(SDXL_com[image_url][-2])/len(SDXL_com[image_url][-2])
#     print(f'{image_url}:')
#     print(f'auraflow: {AuraFlow_com[image_url][1]}')
#     print(f'sdxl: {SDXL_com[image_url][1]}')
#     for alpha in alphas:
#         AIM_auraflow = (auraflow_action_score * 1 + auraflow_reason_score * (alpha)) / (1 + alpha)
#         AIM_sdxl = (sdxl_action_score * 1 + sdxl_reason_score * (alpha)) / (1 + alpha)
#         winner = 'auraflow' if AIM_auraflow > AIM_sdxl else 'sdxl'
#         print(f'alpha: {alpha}, winner: {winner}')
#     print('--------------------------------')

# print('--------------PSA----------------')
# for image_url in PSA:
#     if AuraFlow_psa[image_url][0] == 0:
#         auraflow_action_score = 0
#         auraflow_reason_score = 0
#     else:
#         auraflow_reason_score = sum(AuraFlow_psa[image_url][-1])/len(AuraFlow_psa[image_url][-1])
#         auraflow_action_score = sum(AuraFlow_psa[image_url][-2])/len(AuraFlow_psa[image_url][-2])
#     if SDXL_psa[image_url][0] == 0:
#         sdxl_action_score = 0
#         sdxl_reason_score = 0
#     else:
#         sdxl_reason_score = sum(SDXL_psa[image_url][-1])/len(SDXL_psa[image_url][-1])
#         sdxl_action_score = sum(SDXL_psa[image_url][-2])/len(SDXL_psa[image_url][-2])
#     print(f'{image_url}:')
#     print(f'auraflow: {AuraFlow_psa[image_url][1]}')
#     print(f'sdxl: {SDXL_psa[image_url][1]}')
#     for alpha in alphas:
#         AIM_auraflow = (auraflow_action_score * 1 + auraflow_reason_score * (alpha)) / (1 + alpha)
#         AIM_sdxl = (sdxl_action_score * 1 + sdxl_reason_score * (alpha)) / (1 + alpha)
#         winner = 'auraflow' if AIM_auraflow > AIM_sdxl else 'sdxl'
#         print(f'alpha: {alpha}, winner: {winner}')
#     print('--------------------------------')


import json 

data_no_object = json.load(open('/Users/aysanaghazadeh/experiments/results/IN_InternVL_LLAMA3Instruct_descriptions_AuraFlow_20240817_185858_description_single_paragraph_full_description_persuasion_creativity_reason_only.json'))
reason_no_object = json.load(open('/Users/aysanaghazadeh/experiments/results/IN_InternVL_LLAMA3Instruct_descriptions_AuraFlow_20240817_185858_description_single_paragraph_no_textLLAMA3_instruct_text_image_alignment_isFineTunedTrue_3000_weighted.json'))
data_with_object = json.load(open('/Users/aysanaghazadeh/experiments/results/IN_InternVL_LLM_input_objects_LLAMA3_instruct_FTFalse_COM_AuraFlow_20250810_225032_description_single_paragraph_full_description_with_text_LLAMA3_instruct_persuasion_creativity.json'))
reason_with_object = json.load(open('/Users/aysanaghazadeh/experiments/results/IN_InternVL_LLM_input_objects_LLAMA3_instruct_FTFalse_COM_AuraFlow_20250810_225032_description_single_paragraph_full_description_with_textLLAMA3_instruct_text_image_alignment_isFineTunedTrue_3000_weighted.json'))
count = 0
no_object_reason_score = 0
with_object_reason_score = 0
no_object_data_score = 0
with_object_data_score = 0
no_object_PA_score = 0
with_object_PA_score = 0
AIM_no_object = 0
AIM_with_object = 0
for image_url in data_no_object:
    if image_url not in data_with_object or image_url not in reason_with_object or image_url not in reason_no_object:
        continue
    no_object_reason_score += sum(reason_no_object[image_url][-1])/len(reason_no_object[image_url][-1])
    with_object_reason_score += sum(reason_with_object[image_url][-1])/len(reason_with_object[image_url][-1])
    no_object_data_score += sum(data_no_object[image_url][:-1])/len(data_no_object[image_url][:-1])/5
    with_object_data_score += sum(data_with_object[image_url][:-1])/len(data_with_object[image_url][:-1])/5
    no_object_PA =(sum(data_no_object[image_url][:-1]) / 5 + sum(reason_no_object[image_url][-1])/len(reason_no_object[image_url][-1])) / 8
    with_object_PA =(sum(data_with_object[image_url][:-1]) / 5 + sum(reason_with_object[image_url][-1])/len(reason_with_object[image_url][-1])) / 8
    no_object_PA_score += no_object_PA
    with_object_PA_score += with_object_PA
    AIM_no_object += reason_no_object[image_url][1]
    AIM_with_object += reason_with_object[image_url][1]
    count += 1
print(f'no object reason score: {no_object_reason_score/count}')
print(f'with object reason score: {with_object_reason_score/count}')
print(f'no object data score: {no_object_data_score/count}')
print(f'with object data score: {with_object_data_score/count}')
print(f'no object PA score: {no_object_PA_score/count}')
print(f'with object PA score: {with_object_PA_score/count}')
print(f'AIM no object: {AIM_no_object/count}')
print(f'AIM with object: {AIM_with_object/count}')