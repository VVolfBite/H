from typing import Optional, ClassVar, List

from langchain.llms.base import LLM
from pydantic import PrivateAttr
import threading
import websocket
import json
import ssl
import hmac
import hashlib
import base64
from urllib.parse import urlparse, urlencode
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread


class Ws_Param:
    def __init__(self, APPID, APIKey, APISecret, gpt_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(gpt_url).netloc
        self.path = urlparse(gpt_url).path
        self.gpt_url = gpt_url

    def create_url(self):
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        signature_origin = f"host: {self.host}\ndate: {date}\nGET {self.path} HTTP/1.1"
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha_base64 = base64.b64encode(signature_sha).decode('utf-8')
        authorization_origin = (
            f'api_key="{self.APIKey}", algorithm="hmac-sha256", '
            f'headers="host date request-line", signature="{signature_sha_base64}"'
        )
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')

        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        url = self.gpt_url + "?" + urlencode(v)
        return url


class RemoteLLM(LLM):
    # 固定参数作为类变量
    APPID: ClassVar[str] = "4b69b0da"
    API_KEY: ClassVar[str] = "44d76e7af568905bfaf6e7c1b332a98a"
    API_SECRET: ClassVar[str] = "YmRlZTc0ZTkxNDQ5ZWE2Njk2ZTUwYTFi"
    GPT_URL: ClassVar[str] = "wss://spark-api.xf-yun.com/v1.1/chat"
    DOMAIN: ClassVar[str] = "lite"

    # 私有属性，状态变量，不被Pydantic管理
    _lock: threading.Event = PrivateAttr()
    _ws: Optional[websocket.WebSocketApp] = PrivateAttr(default=None)
    _result: Optional[str] = PrivateAttr(default=None)
    _query: Optional[str] = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lock = threading.Event()
        self._ws = None
        self._result = None
        self._query = None

    def _get_ws_param(self):
        return Ws_Param(self.APPID, self.API_KEY, self.API_SECRET, self.GPT_URL)

    def _gen_params(self, appid, query, domain):
        return {
            "header": {
                "app_id": appid,
                "uid": "1234",
            },
            "parameter": {
                "chat": {
                    "domain": domain,
                    "temperature": 0.0,
                    "max_tokens": 1000,
                    "auditing": "default",
                }
            },
            "payload": {
                "message": {
                    "text": [{"role": "user", "content": query}]
                }
            }
        }

    def _on_message(self, ws, message):
        data = json.loads(message)
        code = data.get('header', {}).get('code', -1)
        if code != 0:
            print(f"请求错误: {code}, {data}")
            ws.close()
            self._result = None
            self._lock.set()
            return

        payload = data.get("payload", {})
        choices = payload.get("choices", {})
        status = choices.get("status", -1)
        content_list = choices.get("text", [])

        if content_list:
            content = content_list[0].get("content", "")
            if self._result is None:
                self._result = content
            else:
                self._result += content

        if status == 2:  # 结束
            ws.close()
            self._lock.set()

    def _on_error(self, ws, error):
        print("### error:", error)
        self._result = None
        self._lock.set()

    def _on_close(self, ws, close_status_code, close_msg):
        self._lock.set()

    def _on_open(self, ws):
        def run(*args):
            params = self._gen_params(self.APPID, self._query, self.DOMAIN)
            ws.send(json.dumps(params))
        thread.start_new_thread(run, ())

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        self._query = prompt
        self._result = None
        self._lock.clear()

        ws_param = self._get_ws_param()
        ws_url = ws_param.create_url()

        self._ws = websocket.WebSocketApp(ws_url,
                                          on_message=self._on_message,
                                          on_error=self._on_error,
                                          on_close=self._on_close,
                                          on_open=self._on_open)
        # 开新线程跑websocket长连接
        thread_ws = threading.Thread(target=lambda: self._ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE}))
        thread_ws.daemon = True
        thread_ws.start()

        # 阻塞等待结果，最长等待15秒
        if not self._lock.wait(15):
            print("请求超时，关闭连接")
            self._ws.close()
            thread_ws.join()

        return self._result or ""

    @property
    def _llm_type(self) -> str:
        return "remote"


if __name__ == "__main__":
    llm = RemoteLLM()

    prompt = "请简述循环神经网络的优点。"
    print("Prompt:", prompt)
    answer = llm.invoke(prompt)
    print("回答:", answer)
