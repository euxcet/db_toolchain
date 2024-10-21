import requests

def get_character(path: list):

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
                    "writing_area_width": 425,
                    "writing_area_height": 194,
                },
                "pre_context": "",
                "max_num_results": 10,
                "max_completions": 0,
                "language": "en",
                "ink": [
                    [[327,326,326,326,326,326,326,326,326,326,325,325,325,324,324,324,323,323,323,322,322,321,320,320,319,319,319,318,317,316,316,315,315,314,314,313,312,312,312,311,311,310,310,309,309,307,307,306,306,305,304,304,303,303,302,301,301,300,299,296,296,295,294,294,293,293,292,291,291,291,288,287,287,286,285,285,285,284,283,283,283,282,282,281,281,280,280,279,279,278,278,277,277,276,276,275,274,274,273,273,273,272,272,272,271,271,271,271,271,271,270,270,270,270,270,270,270,270,271,271,272,272,272,273,273,274,274,275,275,278,279,280,282,282,283,284,284,286,286,286,288,289,289,291,291,293,294,295,296,298,299,300,302,303,304,306,308,308,309,311,313,314,316,319,322,322,324,326,327,329,332,334,335,337,338,340,342,343,344,345],
                    [125,125,124,124,124,124,124,124,123,123,123,123,123,122,122,122,121,121,121,121,121,120,120,120,120,119,119,119,118,118,118,118,117,117,117,117,117,117,117,117,116,116,116,116,116,116,116,116,116,116,116,116,116,116,116,116,116,116,116,117,117,118,118,118,118,119,119,119,120,120,121,122,122,123,123,123,124,124,125,125,125,126,126,127,127,128,128,129,129,130,130,131,132,133,133,134,136,137,137,138,139,139,140,142,143,144,145,146,147,148,149,150,151,151,152,153,154,156,157,158,159,160,161,161,162,162,163,163,164,167,167,169,169,170,170,171,171,172,172,172,173,173,174,175,175,175,176,176,176,177,177,178,178,179,179,179,180,180,180,180,180,181,181,181,181,181,182,182,182,182,182,182,182,182,182,182,182,182,182,182],
                    [0,9,12,23,26,27,28,30,30,32,33,33,35,37,37,39,39,41,42,43,44,45,46,47,47,48,50,50,53,55,56,57,58,59,59,61,62,63,64,65,65,66,68,70,71,74,75,75,77,78,79,80,81,82,83,84,85,87,88,91,92,92,94,94,96,97,97,98,100,100,101,104,105,106,107,108,108,110,110,112,112,114,114,115,117,117,120,121,122,123,124,124,126,127,128,129,130,131,131,133,134,135,136,138,138,139,141,142,142,144,145,146,147,148,149,150,151,154,155,158,159,159,161,162,163,165,166,167,168,174,175,176,178,179,180,181,182,184,185,186,188,189,190,192,193,195,196,197,198,200,201,202,204,206,207,209,211,212,213,215,217,218,220,224,225,227,230,232,233,235,237,239,240,243,244,246,247,249,251,253],
                    ],
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

    print(response.text)
    # save to file
    with open("post_to_google.txt", "w") as f:
        f.write(response.text)

if __name__ == '__main__':
    get_character([])