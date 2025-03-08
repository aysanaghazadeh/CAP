import json
import os.path


first_image_list = [
    "6/143816.jpg",
    "8/7608.jpg",
    "4/102794.jpg",
    "4/134044.jpg",
    "6/17896.jpg",
    "7/116347.jpg",
    "7/46267.jpg",
    "6/90766.jpg",
    "6/17276.jpg",
    "3/134973.jpg",
    "2/116492.jpg",
    "0/73110.jpg",
    "0/45970.jpg",
    "0/41530.jpg"
]


second_image_list = [
    "0/41530.jpg",
    "5/71965.jpg",
    "0/116140.jpg",
    "1/57011.jpg",
    "3/134033.jpg",
    "5/71815.jpg",
    "1/112811.jpg",
    "3/81633.jpg",
    "4/134044.jpg",
    "6/143816.jpg",
    "7/41457.jpg"
]

third_image_list = [
    "8/138668.jpg", "8/138668.jpg", "8/138668.jpg",
    "8/121068.jpg", "8/121068.jpg", "8/121068.jpg",
    "8/57788.jpg", "8/57788.jpg", "8/57788.jpg",
    "7/33807.jpg", "7/33807.jpg", "7/33807.jpg",
    "4/129394.jpg", "4/129394.jpg", "4/129394.jpg",
    "4/78954.jpg", "4/78954.jpg", "4/78954.jpg",
    "4/46914.jpg", "4/46914.jpg", "4/46914.jpg",
    "2/146302.jpg", "2/146302.jpg", "2/146302.jpg",
    "2/124992.jpg", "2/124992.jpg", "2/124992.jpg",
    "2/57982.jpg", "2/57982.jpg", "2/57982.jpg",
    "0/111880.jpg", "0/111880.jpg", "0/111880.jpg",
    "0/24680.jpg", "0/24680.jpg", "0/24680.jpg",
    "8/52488.jpg", "8/52488.jpg", "8/52488.jpg",
    "4/84094.jpg", "4/84094.jpg", "4/84094.jpg",
    "4/46914.jpg", "4/46914.jpg", "4/46914.jpg",
    "6/14807",
]

fourth_list = ["8/52488.jpg", "8/52488.jpg", "8/52488.jpg",
    "4/84094.jpg", "4/84094.jpg", "4/84094.jpg",
    "4/46914.jpg", "4/46914.jpg", "4/46914.jpg",
    "6/148076.jpg", "6/148076.jpg", "6/148076.jpg",
    "6/158126.jpg", "6/158126.jpg", "6/158126.jpg",
    "8/121068.jpg", "8/121068.jpg", "8/121068.jpg",
    "6/155826.jpg", "6/155826.jpg", "6/155826.jpg",
    "5/129255.jpg", "5/129255.jpg", "5/129255.jpg",
    "7/12117.jpg", "7/12117.jpg", "7/12117.jpg",
    "5/51325.jpg", "5/51325.jpg", "5/51325.jpg",
    "6/125426.jpg", "6/125426.jpg", "6/125426.jpg",
    "7/54357.jpg", "7/54357.jpg", "7/54357.jpg",
    "5/107845.jpg", "5/107845.jpg", "5/107845.jpg",
    "5/128995.jpg", "5/128995.jpg", "5/128995.jpg",
    "7/115587.jpg", "7/115587.jpg", "7/115587.jpg",
    "6/141276.jpg", "6/141276.jpg", "6/141276.jpg"]



while True:
    try:
        AIM_path = os.path.join('../experiments/results', input('AIM path:'))
        PA_path = os.path.join('../experiments/results', input('PA path:'))
        image_set = input('which set:')
        if image_set == 'first':
            image_list = first_image_list
        elif image_set == 'third':
            image_list = third_image_list
        elif image_set == 'second':
            image_list = second_image_list
        else:
            image_list = fourth_list
        AIM = json.load(open(AIM_path))

        if PA_path == None:
            sum_val, image_count = 0, 0
            for image_url in AIM:
                sum_val += AIM[image_url]
                image_count += 1
            print(sum_val/image_count)
            continue
        pa = json.load(open(PA_path))
        sum_val, image_count = 0, 0
        for image_url in AIM:
            if image_url not in pa:
                continue
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
            if image_url in image_list:
                print(f'score for {image_url}is {score/count}')
                print(f'score for {image_url}is {AIM[image_url][1]}')

        print(sum_val / image_count, image_count)
        print('-' * 150)
    except Exception as e:
        # Code to handle the exception
        print("An error occurred:", e)
        print(AIM[image_url])
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected! Exiting...")
        break