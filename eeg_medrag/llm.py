"""OpenAI Chat Completions and embedding callbacks."""

from __future__ import annotations

import os
import logging
from time import perf_counter
from collections.abc import Awaitable, Callable
from typing import Any
from functools import partial

import numpy as np
from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    AsyncOpenAI,
    NOT_GIVEN,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from tiktoken import encoding_for_model
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ._vendor.hypergraphrag.utils import limit_async_func_call
from .base import LLMConfig
from .request_usage import record_request, request_namespace
from .utils import IntegrityError, component_descriptor, identity


logger = logging.getLogger(__name__)

CHAT_MODEL_IDS = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gemini-2.5-flash": "gemini-2.5-flash-lite",
    "deepseek-v4-flash": "deepseek-v4-1-flash-economy-261017",
}

def _usage(response) -> dict | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    return {k: getattr(usage, k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            if hasattr(usage, k)}


def _log_retry(state: RetryCallState) -> None:
    error = state.outcome.exception() if state.outcome is not None else None
    logger.warning(
        "Model retry attempt=%d delay_seconds=%.3f error=%s http_status=%s request_id=%s",
        state.attempt_number, state.next_action.sleep if state.next_action else 0.0,
        type(error).__name__, getattr(error, "status_code", None),
        getattr(error, "request_id", None),
    )


def _retryable(error: BaseException) -> bool:
    if isinstance(error, APIStatusError):
        return error.status_code in (408, 409, 425, 429) or error.status_code >= 500
    return isinstance(
        error, (APIConnectionError, APIResponseValidationError, IntegrityError)
    )


def _attempts(count: int, wait: float, maximum: float, retryable=_retryable) -> AsyncRetrying:
    return AsyncRetrying(
        stop=stop_after_attempt(count),
        wait=wait_exponential(multiplier=1 if wait else 0, min=wait, max=maximum),
        retry=retry_if_exception(retryable),
        reraise=True,
        before_sleep=_log_retry,
    )


class ContentFilteredAnswer(IntegrityError):
    """The service explicitly filtered an empty completion."""


def _completion_text(response: ChatCompletion) -> str:
    try:
        choice = response.choices[0]
        if len(response.choices) == 1 and choice.finish_reason == "content_filter":
            content = choice.message.content
            if content is None or (isinstance(content, str) and not content.strip()):
                raise ContentFilteredAnswer("Empty completion: finish_reason='content_filter'")
        if choice.finish_reason != "stop":
            raise IntegrityError(
                f"Incomplete completion: finish_reason={choice.finish_reason!r}"
            )
        if getattr(choice.message, "refusal", None):
            raise IntegrityError("Provider refused completion")
        text = choice.message.content
        if not (type(text) is str and text.strip()):
            raise IntegrityError("Empty completion")
        return text
    except (AttributeError, IndexError, TypeError) as error:
        raise IntegrityError("Malformed completion response") from error


async def openai_complete(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[ChatCompletionMessageParam] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs,
) -> Any:
    """Request a completion with validation and bounded retries.

    Authentication, endpoint and omitted timeout values follow the SDK
    environment and default settings.
    """
    started = perf_counter()
    validate = kwargs.pop("validator", None)
    count = kwargs.pop("max_attempts", LLMConfig.max_attempts)
    wait = kwargs.pop("backoff_seconds", LLMConfig.backoff_seconds)
    timeout = kwargs.pop("timeout", None)
    timeout = NOT_GIVEN if timeout is None else timeout
    messages: list[ChatCompletionMessageParam] = (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    )
    messages.extend(history_messages or [])
    messages.append({"role": "user", "content": prompt})
    original_messages = list(messages)
    answering = request_namespace() == "answer_generation"
    previous_failure = None

    def retry_chat_error(error: BaseException) -> bool:
        nonlocal previous_failure
        if not answering:
            return _retryable(error)
        if not isinstance(error, ContentFilteredAnswer):
            previous_failure = error
        return isinstance(error, Exception)

    async with AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout) as client:
        async for attempt in _attempts(count, wait, 10, retry_chat_error):
            with attempt:
                call_started = perf_counter()
                call = {"event": "sdk_call", "operation": "chat", "requested_model": model,
                        "attempt": attempt.retry_state.attempt_number}
                try:
                    response = await client.chat.completions.create(
                        model=model, messages=messages, stream=False, **kwargs
                    )
                except Exception as error:
                    record_request(**call, status="error", elapsed_seconds=perf_counter() - call_started,
                                   error_type=type(error).__name__, usage=None,
                                   http_status=getattr(error, "status_code", None),
                                   request_id=getattr(error, "request_id", None))
                    raise
                usage = getattr(response, "usage", None)
                call.update(elapsed_seconds=perf_counter() - call_started,
                            response_model=getattr(response, "model", None),
                            request_id=getattr(response, "_request_id", None),
                            usage=_usage(response))
                try:
                    text = _completion_text(response)
                    output_limit = kwargs.get("max_completion_tokens", kwargs.get("max_tokens"))
                    output_tokens = getattr(usage, "completion_tokens", None)
                    if (isinstance(output_limit, int) and isinstance(output_tokens, int)
                            and output_tokens > output_limit):
                        raise IntegrityError(
                            f"Completion used {output_tokens} tokens, exceeding the requested limit of {output_limit}"
                        )
                    try:
                        value = validate(text) if validate is not None else text
                    except (ValueError, TypeError, KeyError, IndexError) as error:
                        raise IntegrityError(
                            "Invalid completion output: " + str(error)
                        ) from error
                except IntegrityError as error:
                    if not (answering and isinstance(error, ContentFilteredAnswer)):
                        messages = original_messages + [{
                            "role": "user",
                            "content": (
                                "The previous response failed output validation: " + str(error)
                                + "\nGenerate a complete replacement for the original task, following "
                                "its output format and constraints. Do not explain the formatting "
                                "error. Do not change the task criteria to make validation pass."
                            ),
                        }]
                    choices = getattr(response, "choices", None) or []
                    choice = choices[0] if choices else None
                    record_request(**call, status="invalid", error_type=type(error).__name__,
                                   finish_reason=getattr(choice, "finish_reason", None))
                    logger.warning(
                        "Chat response rejected model=%s attempt=%d finish_reason=%s "
                        "prompt_tokens=%s completion_tokens=%s max_tokens=%s request_id=%s",
                        model, attempt.retry_state.attempt_number,
                        getattr(choice, "finish_reason", None),
                        getattr(usage, "prompt_tokens", None),
                        getattr(usage, "completion_tokens", None),
                        kwargs.get("max_tokens"), getattr(response, "_request_id", None),
                    )
                    if (isinstance(error, ContentFilteredAnswer) and previous_failure is not None
                            and attempt.retry_state.attempt_number == count):
                        raise previous_failure from error
                    raise
                record_request(**call, status="success", finish_reason="stop")
                logger.info(
                    "Chat completed model=%s attempts=%d elapsed_seconds=%.3f prompt_tokens=%s completion_tokens=%s request_id=%s",
                    model, attempt.retry_state.attempt_number, perf_counter() - started,
                    getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None),
                    getattr(response, "_request_id", None),
                )
                return value


async def gpt_4o_mini_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[ChatCompletionMessageParam] | None = None,
    **kwargs,
) -> Any:
    return await openai_complete(
        CHAT_MODEL_IDS["gpt-4o-mini"],
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gemini_2_5_flash_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[ChatCompletionMessageParam] | None = None,
    **kwargs,
) -> Any:
    return await openai_complete(
        CHAT_MODEL_IDS["gemini-2.5-flash"],
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def deepseek_v4_flash_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[ChatCompletionMessageParam] | None = None,
    **kwargs,
) -> Any:
    return await openai_complete(
        CHAT_MODEL_IDS["deepseek-v4-flash"],
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def openai_embedding(
    texts: list[str],
    model: str = LLMConfig.embedding_model,
    base_url: str | None = None,
    api_key: str | None = None,
    *,
    dimensions: int = LLMConfig.embedding_dimension,
    max_token_size: int = LLMConfig.embedding_max_token_size,
    **kwargs,
) -> np.ndarray:  # type: ignore[return]  # Retry attempts return or raise.
    """Check input lengths, request embeddings and validate returned vectors."""
    started = perf_counter()
    input_tokens = 0
    encoder = encoding_for_model(model)
    for index, text in enumerate(texts):
        tokens = len(encoder.encode(text, disallowed_special=()))
        input_tokens += tokens
        if tokens > max_token_size:
            logger.error(
                "Embedding input exceeds limit model=%s index=%d tokens=%d limit=%d",
                model, index, tokens, max_token_size,
            )
            raise ValueError(
                f"Embedding input at index {index} has {tokens} tokens; "
                f"limit is {max_token_size} for {model}"
            )
    count = kwargs.pop("max_attempts", LLMConfig.max_attempts)
    wait = kwargs.pop("backoff_seconds", LLMConfig.backoff_seconds)
    timeout = kwargs.pop("timeout", None)
    timeout = NOT_GIVEN if timeout is None else timeout
    async with AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout) as client:
        async for attempt in _attempts(count, wait, 60):
            with attempt:
                call_started = perf_counter()
                call = {"event": "sdk_call", "operation": "embedding", "requested_model": model,
                        "attempt": attempt.retry_state.attempt_number, "inputs": len(texts)}
                try:
                    response = await client.embeddings.create(
                        model=model, input=texts, dimensions=dimensions,
                        encoding_format="float", **kwargs
                    )
                except Exception as error:
                    record_request(**call, status="error", elapsed_seconds=perf_counter() - call_started,
                                   error_type=type(error).__name__, usage=None,
                                   http_status=getattr(error, "status_code", None),
                                   request_id=getattr(error, "request_id", None))
                    raise
                call.update(elapsed_seconds=perf_counter() - call_started,
                            response_model=getattr(response, "model", None), usage=_usage(response),
                            request_id=getattr(response, "_request_id", None))
                try:
                    rows = response.data
                    if not (type(rows) is list and len(rows) == len(texts)):
                        raise IntegrityError("Embedding response count mismatch")
                    if not (
                        all((type(row.index) is int for row in rows))
                        and sorted((row.index for row in rows))
                        == list(range(len(texts)))
                    ):
                        raise IntegrityError("Embedding indexes missing/duplicated")
                    result = np.asarray(
                        [row.embedding for row in sorted(rows, key=lambda r: r.index)]
                    )
                    if not (
                        result.shape == (len(texts), dimensions)
                        and np.isfinite(result).all()
                        and (
                            np.linalg.norm(result.astype(np.float64), axis=1) > 0
                        ).all()
                    ):
                        raise IntegrityError(
                            "Invalid embedding dimensions/nonfinite/zero values"
                        )
                except (AttributeError, IndexError, TypeError, ValueError) as error:
                    record_request(**call, status="invalid", error_type=type(error).__name__)
                    raise IntegrityError(
                        "Malformed embedding response: " + str(error)
                    ) from error
                record_request(**call, status="success")
                logger.info(
                    "Embedding completed model=%s inputs=%d tokens=%d dimensions=%d attempts=%d elapsed_seconds=%.3f request_id=%s",
                    model, len(texts), input_tokens, dimensions,
                    attempt.retry_state.attempt_number, perf_counter() - started,
                    getattr(response, "_request_id", None),
                )
                return result


class LLMProvider:
    """Adapt callbacks with independent completion and embedding concurrency limits."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        completion_func: Callable[..., Awaitable[Any]] | None = None,
        embedding_func: Callable[..., Awaitable[np.ndarray]] = openai_embedding,
    ) -> None:
        self.config = config or LLMConfig()
        self.completion_func = completion_func or {
            "gpt-4o-mini": gpt_4o_mini_complete,
            "gemini-2.5-flash": gemini_2_5_flash_complete,
            "deepseek-v4-flash": deepseek_v4_flash_complete,
        }.get(
            self.config.chat_model,
            partial(openai_complete, self.config.chat_model),
        )
        self.embedding_func = embedding_func
        self._endpoint = os.environ.get("OPENAI_BASE_URL")
        common = {
            "client_library": "openai-sdk",
            "base_url": self._endpoint,
        }
        self.descriptor = {
            **common,
            "chat_model": self.config.chat_model,
            "requested_model": CHAT_MODEL_IDS.get(self.config.chat_model, self.config.chat_model)
            if completion_func is None else None,
            "embedding_model": self.config.embedding_model,
            "embedding_dimension": self.config.embedding_dimension,
        }
        self.chat_signature = identity(
            [
                common,
                component_descriptor(self.completion_func),
                self.config.chat_model,
            ]
        )
        self.embedding_signature = identity(
            [
                common,
                component_descriptor(self.embedding_func),
                self.config.embedding_model,
                self.config.embedding_dimension,
            ]
        )
        self.signature = identity([self.chat_signature, self.embedding_signature])
        self._limited_complete = limit_async_func_call(self.config.concurrency)(
            self.completion_func
        )
        self._limited_embed = limit_async_func_call(self.config.concurrency)(
            self.embedding_func
        )

    async def complete(
        self, messages: list[dict], validator: Callable[[str], Any] | None,
        **parameters,
    ) -> Any:
        system = messages[0]["content"] if messages[0]["role"] == "system" else None
        history = messages[1:-1] if system is not None else messages[:-1]
        return await self._limited_complete(
            messages[-1]["content"],
            system_prompt=system,
            history_messages=history,
            validator=validator,
            base_url=self._endpoint,
            max_attempts=self.config.max_attempts,
            backoff_seconds=self.config.backoff_seconds,
            **parameters,
        )

    async def embed(self, texts: list[str]) -> np.ndarray:
        return await self._limited_embed(
            texts,
            model=self.config.embedding_model,
            dimensions=self.config.embedding_dimension,
            max_token_size=self.config.embedding_max_token_size,
            base_url=self._endpoint,
            max_attempts=self.config.max_attempts,
            backoff_seconds=self.config.backoff_seconds,
        )
