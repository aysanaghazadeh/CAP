import json
import os.path

while True:
    AIM_path = os.path.join('../experiments/results', input('AIM path:'))
    PA_path = os.path.join('../experiments/results', input('PA path:'))
    AIM = json.load(open(AIM_path))
    pa = json.load(open(PA_path))
    sum_val, image_count = 0, 0
    for image_url in AIM:
        # AIM = sum(LLAMA_PSA_AIM[image_url][-1])/len(LLAMA_PSA_AIM[image_url][-1]) if LLAMA_PSA_AIM[image_url][-1] != 0 else 0
        aim_score = AIM[image_url][-1][0] if AIM[image_url][-1] != 0 else 0
        if aim_score > 1:
            print(aim_score, image_url)
        PA = pa[image_url]
        score, count = 0, 0
        for i, value in enumerate(PA):
            if i in [1, -3]:
                continue
            score += value / 5
            count += 1
        score += aim_score * 2
        count += 2
        # print(score/count, count)
        sum_val += score / count
        image_count += 1
    print(sum_val / image_count, image_count)
    print('-' * 150)