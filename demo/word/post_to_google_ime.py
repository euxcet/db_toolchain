import os
import time
import numpy as np
import requests
import json
from PIL import Image, ImageDraw
import flask

def save_image(data: np.ndarray, width: int, height: int) -> None:
    os.makedirs('result', exist_ok=True)
    canvas = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for i in range(0, data.shape[1] - 1):
        draw.line([(data[0][i], data[1][i]), (data[0][i + 1], data[1][i + 1])], fill=(255, 255, 255), width=1)
    canvas.show()
    canvas.save('result/' + str(time.time()) + '.jpg')

def get_character(data: list):
    SCALE = 1
    PADDING = 20
    data: np.ndarray = np.array(data).T[:3]
    # time
    data[2] = (data[2] - data[2][0]) / 1000
    # x, y
    data[0] = data[0] * SCALE
    data[1] = data[1] * SCALE
    data = data.astype(int)
    data[0] = data[0] - (data[0].min() - PADDING)
    data[1] = data[1] - (data[1].min() - PADDING)
    width = int(data[0].max() + PADDING)
    height = int(data[1].max() + PADDING)

    save_image(data, width, height)

    cookies = {
        "HSID": "A_PlfHP_YK9-9a36P",
        "SSID": "Av7gE6Ua2hbiVa94y",
        "APISID": "cEwJJb0HHdQ5mLRr/AzcgyfwiYC1PInhDO",
        "SAPISID": "2jbQarUxp6jLZXz7/A6jWu8qxXzHvUo4Xt",
        "__Secure-1PAPISID": "2jbQarUxp6jLZXz7/A6jWu8qxXzHvUo4Xt",
        "__Secure-3PAPISID": "2jbQarUxp6jLZXz7/A6jWu8qxXzHvUo4Xt",
        "SEARCH_SAMESITE": "CgQIv5sB",
        "SID": "g.a000nQg33ci48YYqI2By-F4fIdsob-BxulAYikIh_yGL0ybxp3_fo5zngmxSn0mts1vnlZk88gACgYKAfcSAQASFQHGX2Mi3hTtuYkAixl-BvPEKkJKVRoVAUF8yKpKO3QMpTJSTENtchm7GRaF0076",
        "__Secure-1PSID": "g.a000nQg33ci48YYqI2By-F4fIdsob-BxulAYikIh_yGL0ybxp3_fO2Kvq7oN_L3C104AUnk9ywACgYKAVoSAQASFQHGX2Mis6pF-6kalyPoS0KSHiKFjRoVAUF8yKrjN3oNPVkY4pJh5jBJiAj30076",
        "__Secure-3PSID": "g.a000nQg33ci48YYqI2By-F4fIdsob-BxulAYikIh_yGL0ybxp3_fFNcBXfAseqUeh_IKxr3z_AACgYKAfISAQASFQHGX2MiYo0OeItUZfFinuDHaR_1eRoVAUF8yKq2bESZXtErPIXqVdasRW0v0076",
        "OGPC": "19024362-1:",
        "OGP": "-19024362:",
        "__Secure-1PSIDTS": "sidts-CjIBUFGohybIz5uAjvC1ynJ-tEFwsYlE7u6gWk-Ttp3i81-uLB8uXBdebYJlAR8tMLBrDBAA",
        "__Secure-3PSIDTS": "sidts-CjIBUFGohybIz5uAjvC1ynJ-tEFwsYlE7u6gWk-Ttp3i81-uLB8uXBdebYJlAR8tMLBrDBAA",
        "AEC": "AVYB7cotTbH9vTJKkxZU20lQ9QU0hqEdzuuRi75-JfD3YERwTmdfeAGBTQ",
        "NID": "517=mmKmO3QRLNQK51ZMKdr0EUwwBuwmT4N7pjkihZOJTdBecbeAfIMepGU9X77RWRxYs6nyum8U5aD9HoU2KBS1vZBAQkDaiIbV6a2Pt-yMKVTXybpaP29DcfLe1_1QIcT7iJp3LwVwubWu57N-ZSXYcxOJFMSRgFRzb1WWp6kMuy9jh1Xt5MNmefM_CrI53WarTgpuTny_viy3Gh4ajXig_v5UVkRoBDFyJh-FSkzm3gzg8_zOOB988r8EPG_7FlfJ0afZ8W4LbhWOtUy79zDDeAgcWiNKAWyWWXteJWgiPGawqBYBxXiU4ZoxmPwCJLoQBoe-HS5aOhgxEk-l7bdY3EudnQQJHKDV-gJI9GcIT_k6hsZNzreiIXASOvFXymGdYaiZHrx82dSjR6wJHA2cLQKGwtEarGYA5b5ibe4LTJO0r9DTQKn4I7fXAVB9hDrxWS3szv3xaCa1BdGsaVeV45Y",
        "SIDCC": "AKEyXzXGgXYDKqrdPju3MjxZdQ7kLrx8eG8V3n2PvEB7reH2Qg1vmQyaY4FmY0JVQasXkIGft16D",
        "__Secure-1PSIDCC": "AKEyXzWcdYkArle4c17tx_ObdS6ze97Y4-AyK34TqbZzGSWJAsuirasbf68Qz1K9OxxVhIx9aRlm",
        "__Secure-3PSIDCC": "AKEyXzXwHHwIQBwxuvNr8lhfeEONZdkM9yQLJVBoFY8PTc5tEwdSxfs4FnSHB3okRVY9C71cg9oY",
    }

    headers = {
        "accept": "*/*",
        "accept-language": "zh-CN,zh;q=0.9",
        "content-type": "application/json; charset=UTF-8",
        # 'cookie': 'HSID=A_PlfHP_YK9-9a36P; SSID=Av7gE6Ua2hbiVa94y; APISID=cEwJJb0HHdQ5mLRr/AzcgyfwiYC1PInhDO; SAPISID=2jbQarUxp6jLZXz7/A6jWu8qxXzHvUo4Xt; __Secure-1PAPISID=2jbQarUxp6jLZXz7/A6jWu8qxXzHvUo4Xt; __Secure-3PAPISID=2jbQarUxp6jLZXz7/A6jWu8qxXzHvUo4Xt; SEARCH_SAMESITE=CgQIv5sB; SID=g.a000nQg33ci48YYqI2By-F4fIdsob-BxulAYikIh_yGL0ybxp3_fo5zngmxSn0mts1vnlZk88gACgYKAfcSAQASFQHGX2Mi3hTtuYkAixl-BvPEKkJKVRoVAUF8yKpKO3QMpTJSTENtchm7GRaF0076; __Secure-1PSID=g.a000nQg33ci48YYqI2By-F4fIdsob-BxulAYikIh_yGL0ybxp3_fO2Kvq7oN_L3C104AUnk9ywACgYKAVoSAQASFQHGX2Mis6pF-6kalyPoS0KSHiKFjRoVAUF8yKrjN3oNPVkY4pJh5jBJiAj30076; __Secure-3PSID=g.a000nQg33ci48YYqI2By-F4fIdsob-BxulAYikIh_yGL0ybxp3_fFNcBXfAseqUeh_IKxr3z_AACgYKAfISAQASFQHGX2MiYo0OeItUZfFinuDHaR_1eRoVAUF8yKq2bESZXtErPIXqVdasRW0v0076; OGPC=19024362-1:; OGP=-19024362:; __Secure-1PSIDTS=sidts-CjIBUFGohybIz5uAjvC1ynJ-tEFwsYlE7u6gWk-Ttp3i81-uLB8uXBdebYJlAR8tMLBrDBAA; __Secure-3PSIDTS=sidts-CjIBUFGohybIz5uAjvC1ynJ-tEFwsYlE7u6gWk-Ttp3i81-uLB8uXBdebYJlAR8tMLBrDBAA; AEC=AVYB7cotTbH9vTJKkxZU20lQ9QU0hqEdzuuRi75-JfD3YERwTmdfeAGBTQ; NID=517=mmKmO3QRLNQK51ZMKdr0EUwwBuwmT4N7pjkihZOJTdBecbeAfIMepGU9X77RWRxYs6nyum8U5aD9HoU2KBS1vZBAQkDaiIbV6a2Pt-yMKVTXybpaP29DcfLe1_1QIcT7iJp3LwVwubWu57N-ZSXYcxOJFMSRgFRzb1WWp6kMuy9jh1Xt5MNmefM_CrI53WarTgpuTny_viy3Gh4ajXig_v5UVkRoBDFyJh-FSkzm3gzg8_zOOB988r8EPG_7FlfJ0afZ8W4LbhWOtUy79zDDeAgcWiNKAWyWWXteJWgiPGawqBYBxXiU4ZoxmPwCJLoQBoe-HS5aOhgxEk-l7bdY3EudnQQJHKDV-gJI9GcIT_k6hsZNzreiIXASOvFXymGdYaiZHrx82dSjR6wJHA2cLQKGwtEarGYA5b5ibe4LTJO0r9DTQKn4I7fXAVB9hDrxWS3szv3xaCa1BdGsaVeV45Y; SIDCC=AKEyXzXGgXYDKqrdPju3MjxZdQ7kLrx8eG8V3n2PvEB7reH2Qg1vmQyaY4FmY0JVQasXkIGft16D; __Secure-1PSIDCC=AKEyXzWcdYkArle4c17tx_ObdS6ze97Y4-AyK34TqbZzGSWJAsuirasbf68Qz1K9OxxVhIx9aRlm; __Secure-3PSIDCC=AKEyXzXwHHwIQBwxuvNr8lhfeEONZdkM9yQLJVBoFY8PTc5tEwdSxfs4FnSHB3okRVY9C71cg9oY',
        "origin": "https://www.google.com",
        "priority": "u=1, i",
        "referer": "https://www.google.com/",
        "sec-ch-ua": '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "x-client-data": "CIa2yQEIpLbJAQipncoBCJ+FywEIk6HLAQiGoM0BCMKszgEI5K/OAQjDts4BCLy5zgEI173OAQjjvs4BGPbJzQEYnbHOARi/k9Ui",
    }

    params = {
        "itc": "en-t-i0-handwrit",
        "app": "demopage",
    }

    json_data = {
        "itc": "en-t-i0-handwrit",
        "app_version": 0.4,
        "api_level": "537.36",
        "device": "5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "input_type": "0",
        "options": "enable_pre_space",
        "requests": [
            {
                "writing_guide": {
                    "writing_area_width": width,
                    "writing_area_height": height,
                },
                "pre_context": "",
                "max_num_results": 10,
                "max_completions": 0,
                "language": "en",
                "ink": [
                    data.tolist(),
                ],
            },
        ],
    }
    
    response = requests.post(
        "https://inputtools.google.com/request",
        params=params,
        cookies=cookies,
        headers=headers,
        json=json_data,
    )

    print(json.loads(response.text)[1][0][1])

if __name__ == '__main__':
    get_character([[0, 0, 0], [0, 2, 0.04], [0, 4, 0.04]])
