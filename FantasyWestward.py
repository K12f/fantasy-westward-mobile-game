import time
import uuid

import chromadb
import cv2
from adbutils import adb
from playwright.async_api import async_playwright

from paddleocr import PaddleOCR


class FantasyWestward:
    ANSWERS_VECTOR_COLLECTION: str = "smart-answer"
    ANSWERS_VECTOR_DB: str = "answers.txt"

    PADDLEOCR_DET_MODEL_PATH: str = "./paddleocr/whl/det/ch/ch_PP-OCRv4_det_infer"
    PADDLEOCR_REC_MODEL_PATH: str = "./paddleocr/whl/rec/ch/ch_PP-OCRv4_rec_infer"
    PADDLEOCR_CLS_MODEL_PATH: str = "./paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer"

    __INSTANCE = None
    __chroma_client = None
    __answer_collection = None
    __paddle_ocr = None

    def __new__(cls, *args, **kwargs):
        if not cls.__INSTANCE:
            cls.__INSTANCE = super().__new__(cls)
        return cls.__INSTANCE

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        return self

    async def init_answer_vector(self):
        # 打开原始文本文件和新的文本文件
        data_list = []
        ids = []
        with open(self.ANSWERS_VECTOR_DB, "r", encoding="utf-8") as file:
            lines = file.readlines()
            # 为每个数据生成一个新的字典，其中包含使用 UUID 生成的 ID
            for line in lines:
                data_id = str(uuid.uuid4())  # 使用UUID生成唯一的ID
                data = line.strip()  # 去除行尾的换行符等
                ids.append(data_id)
                data_list.append(data)
        self.__chroma_client = chromadb.Client()

        self.__answer_collection = self.__chroma_client.create_collection(self.ANSWERS_VECTOR_COLLECTION)

        self.__answer_collection.add(documents=data_list, ids=ids)

    async def init_paddle_ocr(self):
        self.__paddle_ocr = PaddleOCR(
            det_model_dir=self.PADDLEOCR_DET_MODEL_PATH, rec_model_dir=self.PADDLEOCR_REC_MODEL_PATH,
            cls_model_dir=self.PADDLEOCR_CLS_MODEL_PATH,
            use_angle_cls=True, lang="ch",
            use_gpu=False,
            show_log=False,
        )

    async def baidu_search(self, query, retry=3):
        """
        百度搜索
        :param query:
        :param retry:
        :return:
        """
        data = []
        try:
            while retry > 0:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    context = await browser.new_context()
                    page = await context.new_page()

                    # 打开搜索页面
                    await page.goto("https://www.baidu.com")

                    # 在搜索框中输入查询词
                    await page.type("#kw", query)

                    # 模拟按下Enter键
                    await page.press("#su", "Enter")

                    # 等待搜索结果加载
                    await page.wait_for_selector("#page")

                    # 获取搜索结果页面标题
                    answer_elements = page.locator(
                        "#content_left .c-container .content-right_8Zs40"
                    )
                    for answer_element in await answer_elements.all():
                        answer = await answer_element.inner_text()
                        data.append(answer)
                    await browser.close()
                return data
        except Exception as err:
            print(err)
            retry -= 1
            await self.baidu_search(query, retry)
        return data

    async def screenshot(self, filename: str = "./screenshot.png"):
        """
        截图
        :param host:
        :param port:
        :param filename:
        :return:
        """
        try:
            adb.connect(f"{self.host}:{self.port}", 10)
            d = adb.device()
            # 获取截图
            s = d.screenshot()

            s.save(filename)
        except Exception as err:
            print(f"截图失败,请启动mumu模拟器,在问题诊断->查找adb调试端口,并配置.err：{err}")

    async def record_video(self, seconds=3, filename: str = "./test"):
        """
        adb录制手机视频
        :param host:
        :param port:
        :param seconds:
        :param filename:
        :return:
        """
        adb.connect(f"{self.host}:{self.port}", 10)
        d = adb.device()
        stream = d.shell(f"screenrecord /sdcard/{filename}.mp4", stream=True)
        time.sleep(seconds)  # record for 3 seconds
        with stream:
            stream.send(b"\003")  # send Ctrl+C
            stream.read_until_close()
        d.sync.pull(f"/sdcard/{filename}.mp4", f"{filename}.mp4")  # pulling video

    async def read_image(self, path: str):
        """
        open cv读取图片
        """
        image = cv2.imread(path)

        # 确定要截取的区域的坐标和大小
        image_height, image_width, channels = image.shape
        x, y, width, height = round(image_width * 0.41), round(image_height * 0.27), round(image_width * 0.45), round(
            image_height * 0.138)

        # 使用数组切片截取图像的特定区域
        return image[y:y + height, x:x + width]

    async def paddle_ocr(self, img_path: str):
        result_question = self.__paddle_ocr.ocr(img_path, cls=True)
        if result_question is not None and len(result_question) > 0:
            question = result_question[0][0][1][0]
            return question

    async def query_question(self, question):
        return self.__answer_collection.get(
            where_document={"$contains": question},
            include=["documents"],
        )
