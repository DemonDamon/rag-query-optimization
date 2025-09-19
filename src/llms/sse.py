# Date    : 2024/7/9 20:45
# File    : sse.py
# Desc    : 
# Author  : Damon
# E-mail  : bingzhenli@hotmail.com


import io
import re
from datetime import datetime
from functools import partial
from typing import Any, AsyncIterable, Awaitable, Callable, Coroutine, Iterator, Mapping, Optional, Union

import anyio
from starlette.background import BackgroundTask
from starlette.concurrency import iterate_in_threadpool
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from loguru import logger


class SendTimeoutError(TimeoutError):
    """发送超时错误"""
    pass


class AppStatus:
    """用于monkey-patching uvicorn信号处理器的辅助类"""

    def __init__(self):
        pass

    should_exit = False
    should_exit_event: Optional[anyio.Event] = None

    @staticmethod
    def handle_exit(*args, **kwargs):
        """设置退出标志并通知监听者"""
        AppStatus.should_exit = True
        if AppStatus.should_exit_event is not None:
            AppStatus.should_exit_event.set()
        original_handler(*args, **kwargs)


try:
    from uvicorn.main import Server

    original_handler = Server.handle_exit
    Server.handle_exit = AppStatus.handle_exit

    def unpatch_uvicorn_signal_handler():
        """恢复原始信号处理器，取消monkey-patching"""
        Server.handle_exit = original_handler

except ModuleNotFoundError:
    logger.debug("未使用Uvicorn.")


class ServerSentEvent:
    """使用EventSource协议发送数据的类"""
    def __init__(
            self,
            data: Optional[Any] = None,
            event: Optional[str] = None,
            id: Optional[str] = None,
            retry: Optional[int] = None,
            comment: Optional[str] = None,
            sep: Optional[str] = None,
    ):
        self.data = data
        self.event = event
        self.id = id
        self.retry = retry
        self.comment = comment
        self.DEFAULT_SEPARATOR = "\r\n"
        self.LINE_SEP_EXPR = re.compile(r"\r\n|\r|\n")
        self._sep = sep if sep is not None else self.DEFAULT_SEPARATOR

    def encode(self) -> bytes:
        """编码为字节流"""
        buffer = io.StringIO()
        if self.comment is not None:
            for chunk in self.LINE_SEP_EXPR.split(str(self.comment)):
                buffer.write(f": {chunk}")
                buffer.write(self._sep)

        if self.id is not None:
            buffer.write(self.LINE_SEP_EXPR.sub("", f"id: {self.id}"))
            buffer.write(self._sep)

        if self.event is not None:
            buffer.write(self.LINE_SEP_EXPR.sub("", f"event: {self.event}"))
            buffer.write(self._sep)

        if self.data is not None:
            for chunk in self.LINE_SEP_EXPR.split(str(self.data)):
                buffer.write(f"data: {chunk}")
                buffer.write(self._sep)

        if self.retry is not None:
            if not isinstance(self.retry, int):
                raise TypeError("retry argument must be int")
            buffer.write(f"retry: {self.retry}")
            buffer.write(self._sep)

        buffer.write(self._sep)
        return buffer.getvalue().encode("utf-8")


def ensure_bytes(data: Union[bytes, dict, ServerSentEvent, Any], sep: str) -> bytes:
    """确保数据为字节流格式"""
    if isinstance(data, bytes):
        return data
    elif isinstance(data, ServerSentEvent):
        return data.encode()
    elif isinstance(data, dict):
        return ServerSentEvent(**data, sep=sep).encode()
    else:
        return ServerSentEvent(str(data), sep=sep).encode()


Content = Union[str, bytes, dict, ServerSentEvent]
SyncContentStream = Iterator[Content]
AsyncContentStream = AsyncIterable[Content]
ContentStream = Union[AsyncContentStream, SyncContentStream]


class EventSourceResponse(Response):
    """实现了Server-Sent Event协议的响应类"""
    body_iterator: AsyncContentStream

    DEFAULT_PING_INTERVAL = 15

    def __init__(
            self,
            content: ContentStream,
            status_code: int = 200,
            headers: Optional[Mapping[str, str]] = None,
            media_type: str = "text/event-stream",
            background: Optional[BackgroundTask] = None,
            ping: Optional[int] = None,
            sep: Optional[str] = None,
            ping_message_factory: Optional[Callable[[], ServerSentEvent]] = None,
            data_sender_callable: Optional[Callable[[], Coroutine[None, None, None]]] = None,
            send_timeout: Optional[float] = None,
    ):
        if sep and sep not in ["\r\n", "\r", "\n"]:
            raise ValueError(f"分隔符必须为\\r\\n, \\r, \\n之一，当前值：{sep}")
        self.DEFAULT_SEPARATOR = "\r\n"
        self.sep = sep if sep is not None else self.DEFAULT_SEPARATOR
        self.ping_message_factory = ping_message_factory

        if isinstance(content, AsyncIterable):
            self.body_iterator = content
        else:
            self.body_iterator = iterate_in_threadpool(content)
        self.status_code = status_code
        self.media_type = media_type
        self.background = background
        self.data_sender_callable = data_sender_callable
        self.send_timeout = send_timeout

        _headers = headers or {}
        _headers.setdefault("Cache-Control", "no-cache")
        _headers["Connection"] = "keep-alive"
        _headers["X-Accel-Buffering"] = "no"

        self.init_headers(_headers)

        self.ping_interval = self.DEFAULT_PING_INTERVAL if ping is None else ping
        self.active = True
        self._ping_task = None
        self._send_lock = anyio.Lock()

    @staticmethod
    async def listen_for_disconnect(receive: Receive) -> None:
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                logger.debug("Got event: http.disconnect. Stop streaming.")
                break

    @staticmethod
    async def listen_for_exit_signal() -> None:
        # Check if should_exit was set before anybody started waiting
        if AppStatus.should_exit:
            return

        # Setup an Event
        if AppStatus.should_exit_event is None:
            AppStatus.should_exit_event = anyio.Event()

        # Check if should_exit got set while we set up the event
        if AppStatus.should_exit:
            return

        # Await the event
        await AppStatus.should_exit_event.wait()

    async def stream_response(self, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "http_status": self.status_code,
                "headers": self.raw_headers,
            }
        )
        async for data in self.body_iterator:
            chunk = ensure_bytes(data, self.sep)
            # logger.debug(f"chunk: {chunk.decode()}")
            with anyio.move_on_after(self.send_timeout) as timeout:
                await send(
                    {"type": "http.response.body", "body": chunk, "more_body": True}
                )
            if timeout.cancel_called:
                await self.body_iterator.aclose()
                raise SendTimeoutError()

        async with self._send_lock:
            self.active = False
            await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async with anyio.create_task_group() as task_group:
            # https://trio.readthedocs.io/en/latest/reference-core.html#custom-supervisors
            async def wrap(func: Callable[[], Awaitable[None]]) -> None:
                await func()
                # noinspection PyAsyncCall
                task_group.cancel_scope.cancel()

            task_group.start_soon(wrap, partial(self.stream_response, send))
            task_group.start_soon(wrap, partial(self._ping, send))
            task_group.start_soon(wrap, self.listen_for_exit_signal)

            if self.data_sender_callable:
                task_group.start_soon(self.data_sender_callable)

            await wrap(partial(self.listen_for_disconnect, receive))

        if self.background is not None:  # pragma: no cover, tested in StreamResponse
            await self.background()

    def enable_compression(self, force: bool = False) -> None:
        raise NotImplementedError

    @property
    def ping_interval(self) -> Union[int, float]:
        """Time interval between two ping massages"""
        return self._ping_interval

    @ping_interval.setter
    def ping_interval(self, value: Union[int, float]) -> None:
        """Setter for ping_interval property.

        :param int value: interval in sec between two ping values.
        """

        if not isinstance(value, (int, float)):
            raise TypeError("ping interval must be int")
        if value < 0:
            raise ValueError("ping interval must be greater then 0")

        self._ping_interval = value

    async def _ping(self, send: Send) -> None:
        """定期发送心跳包"""
        while self.active:
            await anyio.sleep(self._ping_interval)
            if self.ping_message_factory:
                assert isinstance(self.ping_message_factory,
                                  Callable)  # type: ignore  # https://github.com/python/mypy/issues/6864
            ping = (
                ServerSentEvent(comment=f"ping - {datetime.now()}").encode()
                if self.ping_message_factory is None
                else ensure_bytes(self.ping_message_factory(), self.sep)
            )
            logger.debug(f"ping: {ping.decode()}")
            async with self._send_lock:
                if self.active:
                    await send(
                        {"type": "http.response.body", "body": ping, "more_body": True}
                    )
