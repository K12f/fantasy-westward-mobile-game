import asyncio
import time

import cv2

from FantasyWestward import FantasyWestward


async def main(host: str, port: int):
    COUNT = 100

    """
    0.初始化，将答案放入矢量库
    1.使用adb截图答题时的图片
    2.opencv获取答案的题目
    3.paddleocr识别图片中的文字内容
    4.首先从矢量库中获取，并使用百度搜索，搜索备用答案
    :return:
    """
    async with FantasyWestward(host, port) as fantasy_west:
        # print('init answer vector db')
        # await fantasy_west.init_answer_vector()

        print('init paddle ocr')
        await fantasy_west.init_paddle_ocr()

        while (COUNT > 0):
            COUNT -= 1
            print('scan')
            screenshot_image = './image/screenshot/screenshot.png'
            await fantasy_west.screenshot(host, port)
            crop_image = await fantasy_west.read_image(screenshot_image)
            img_path = './image/crop/crop_image.jpg'
            cv2.imwrite(img_path, crop_image)

            question = await fantasy_west.paddle_ocr(img_path)
            print(question)
            # question_result = await fantasy_west.query_question(question)
            # print(question_result)
            answers = await fantasy_west.baidu_search(question)
            print(answers)
            for answer in answers:
                print(answer)


if __name__ == '__main__':
    t1 = time.time()
    asyncio.run(main('127.0.0.1', 16384))
    t2 = time.time()
    print(t2 - t1)
