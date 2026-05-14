import { OpenAIApiService } from './openai-core.js';
import logger from '../../utils/logger.js';
import crypto from 'crypto';

const DEEPSEEK_BASE_URL = 'https://api.deepseek.com';
const DEFAULT_RESPONSE_CACHE_TTL_MS = 5 * 60 * 1000;
const MAX_RESPONSE_CACHE_ENTRIES = 200;
const responseCache = new Map();

function getBooleanConfig(value, fallback = false) {
    if (value === undefined || value === null || value === '') {
        return fallback;
    }
    if (typeof value === 'boolean') {
        return value;
    }
    return String(value).trim().toLowerCase() === 'true';
}

function getPositiveIntegerConfig(value, fallback) {
    const parsed = Number(value);
    return Number.isFinite(parsed) && parsed > 0 ? Math.round(parsed) : fallback;
}

function normalizeDeepSeekApiKey(apiKey) {
    return String(apiKey || '').trim().replace(/^Bearer\s+/i, '');
}

function normalizeDeepSeekBaseUrl(baseUrl) {
    return String(baseUrl || DEEPSEEK_BASE_URL).trim().replace(/\/+$/, '');
}

function throwReadableDeepSeekError(error) {
    const status = error.response?.status;
    const upstreamMessage = error.response?.data?.error?.message
        || error.response?.data?.message
        || error.message;

    if (status === 401 || status === 403) {
        const readableError = new Error(
            `DeepSeek authentication failed (${status}). 请检查 DeepSeek API Key 是否正确、未过期、未误填 Bearer 前缀，并确认该账号有 API 权限。${upstreamMessage ? ` Upstream: ${upstreamMessage}` : ''}`
        );
        readableError.status = status;
        readableError.cause = error;
        throw readableError;
    }

    throw error;
}

function stringifyContentPart(part) {
    if (part === null || part === undefined) {
        return '';
    }

    if (typeof part === 'string') {
        return part;
    }

    if (typeof part !== 'object') {
        return String(part);
    }

    if (typeof part.text === 'string') {
        return part.text;
    }

    if (part.type === 'image_url') {
        const imageUrl = typeof part.image_url === 'string'
            ? part.image_url
            : part.image_url?.url;
        return imageUrl ? `[Image: ${imageUrl}]` : '[Image]';
    }

    if (part.type === 'input_audio') {
        return `[Audio Input: ${part.input_audio?.format || 'audio'}]`;
    }

    if (part.type === 'tool_result') {
        return typeof part.content === 'string' ? part.content : JSON.stringify(part.content || {});
    }

    if (part.type === 'tool_use') {
        return `[Tool use: ${part.name || 'unknown'}]`;
    }

    return JSON.stringify(part);
}

function normalizeMessageContent(content) {
    if (content === null || content === undefined || typeof content === 'string') {
        return content;
    }

    if (Array.isArray(content)) {
        return content
            .map(stringifyContentPart)
            .filter(Boolean)
            .join('\n');
    }

    return stringifyContentPart(content);
}

function stableStringify(value) {
    if (value === null || typeof value !== 'object') {
        return JSON.stringify(value);
    }
    if (Array.isArray(value)) {
        return `[${value.map(stableStringify).join(',')}]`;
    }

    const keys = Object.keys(value).sort();
    return `{${keys.map(key => `${JSON.stringify(key)}:${stableStringify(value[key])}`).join(',')}}`;
}

function cloneJson(value) {
    return value === undefined ? value : JSON.parse(JSON.stringify(value));
}

function isCacheableDeepSeekRequest(requestBody = {}) {
    if (requestBody._deepseekBypassResponseCache === true) {
        return false;
    }
    if (requestBody.stream === true) {
        return false;
    }
    if (requestBody.tools || requestBody.tool_choice || requestBody.functions || requestBody.function_call) {
        return false;
    }

    const messages = Array.isArray(requestBody.messages) ? requestBody.messages : [];
    return messages.every(message => {
        if (!message || typeof message !== 'object') {
            return false;
        }
        return typeof message.content === 'string' || message.content === null || message.content === undefined;
    });
}

function hasNonTextMessageContent(requestBody = {}) {
    const messages = Array.isArray(requestBody.messages) ? requestBody.messages : [];
    return messages.some(message => {
        if (!message || typeof message !== 'object') {
            return true;
        }
        const content = message.content;
        if (content === null || content === undefined || typeof content === 'string') {
            return false;
        }
        return true;
    });
}

function getDeepSeekResponseCacheKey(namespace, model, requestBody) {
    const cacheBody = { ...requestBody, model, _cacheNamespace: namespace };
    delete cacheBody._monitorRequestId;
    delete cacheBody._requestBaseUrl;
    delete cacheBody._clientCacheNamespace;
    const raw = stableStringify(cacheBody);
    return crypto.createHash('sha256').update(raw).digest('hex');
}

function getCachedDeepSeekResponse(key) {
    const entry = responseCache.get(key);
    if (!entry) {
        return null;
    }
    if (Date.now() > entry.expiresAt) {
        responseCache.delete(key);
        return null;
    }

    responseCache.delete(key);
    responseCache.set(key, entry);
    return cloneJson(entry.value);
}

function setCachedDeepSeekResponse(key, value, ttlMs) {
    responseCache.set(key, {
        value: cloneJson(value),
        expiresAt: Date.now() + ttlMs
    });

    while (responseCache.size > MAX_RESPONSE_CACHE_ENTRIES) {
        const oldestKey = responseCache.keys().next().value;
        responseCache.delete(oldestKey);
    }
}

export function normalizeDeepSeekChatRequest(requestBody = {}) {
    const normalized = { ...requestBody };

    if (Array.isArray(normalized.messages)) {
        normalized.messages = normalized.messages.map(message => {
            if (!message || typeof message !== 'object') {
                return message;
            }

            const normalizedMessage = { ...message };
            if (Object.prototype.hasOwnProperty.call(normalizedMessage, 'content')) {
                normalizedMessage.content = normalizeMessageContent(normalizedMessage.content);
            }
            return normalizedMessage;
        });
    }

    if (normalized.max_completion_tokens !== undefined) {
        if (normalized.max_tokens === undefined) {
            normalized.max_tokens = normalized.max_completion_tokens;
        }
        delete normalized.max_completion_tokens;
    }

    if (normalized.extra_body && typeof normalized.extra_body === 'object' && !Array.isArray(normalized.extra_body)) {
        Object.entries(normalized.extra_body).forEach(([key, value]) => {
            if (normalized[key] === undefined) {
                normalized[key] = value;
            }
        });
        delete normalized.extra_body;
    }

    delete normalized._deepseekBypassResponseCache;
    delete normalized._clientCacheNamespace;

    return normalized;
}

export class DeepSeekApiService extends OpenAIApiService {
    constructor(config) {
        const apiKey = normalizeDeepSeekApiKey(config.DEEPSEEK_API_KEY);
        if (!apiKey) {
            throw new Error("DeepSeek API Key is required for DeepSeekApiService.");
        }
        // Map DeepSeek credentials into the OpenAI-compatible service shape.
        const deepseekConfig = {
            ...config,
            DEEPSEEK_API_KEY: apiKey,
            OPENAI_API_KEY: apiKey,
            OPENAI_BASE_URL: normalizeDeepSeekBaseUrl(config.DEEPSEEK_BASE_URL),
        };
        super(deepseekConfig);
        this.responseCacheEnabled = getBooleanConfig(config.DEEPSEEK_RESPONSE_CACHE_ENABLED, true);
        this.responseCacheTtlMs = getPositiveIntegerConfig(config.DEEPSEEK_RESPONSE_CACHE_TTL_MS, DEFAULT_RESPONSE_CACHE_TTL_MS);
        this.responseCacheNamespace = crypto
            .createHash('sha256')
            .update(`${deepseekConfig.OPENAI_BASE_URL}:${apiKey}`)
            .digest('hex');
        logger.info(`[DeepSeek] Initialized with base URL: ${deepseekConfig.OPENAI_BASE_URL}`);
    }

    async generateContent(model, requestBody) {
        const bypassCache = requestBody?._deepseekBypassResponseCache === true;
        const clientCacheNamespace = requestBody?._clientCacheNamespace || 'default-client';
        const hasNonTextContent = hasNonTextMessageContent(requestBody);
        const normalizedRequest = normalizeDeepSeekChatRequest(requestBody);
        const cacheable = this.responseCacheEnabled && !bypassCache && !hasNonTextContent && isCacheableDeepSeekRequest(normalizedRequest);
        const cacheKey = cacheable
            ? getDeepSeekResponseCacheKey(`${this.responseCacheNamespace}:${clientCacheNamespace}`, model, normalizedRequest)
            : null;

        try {
            if (cacheKey) {
                const cachedResponse = getCachedDeepSeekResponse(cacheKey);
                if (cachedResponse) {
                    cachedResponse._deepseek_response_cache = { hit: true };
                    logger.info(`[DeepSeek] Response cache hit for model ${model}`);
                    return cachedResponse;
                }
            }

            const response = await super.generateContent(model, normalizedRequest);
            if (cacheKey) {
                setCachedDeepSeekResponse(cacheKey, response, this.responseCacheTtlMs);
            }
            return response;
        } catch (error) {
            throwReadableDeepSeekError(error);
        }
    }

    async *generateContentStream(model, requestBody) {
        try {
            yield* super.generateContentStream(model, normalizeDeepSeekChatRequest(requestBody));
        } catch (error) {
            throwReadableDeepSeekError(error);
        }
    }

    async listModels() {
        try {
            const response = await this.axiosInstance.get('/models');
            return response.data;
        } catch (error) {
            const status = error.response?.status;
            logger.error(`[DeepSeek] Error listing models (Status: ${status}):`, error.message);
            return {
                object: 'list',
                data: [
                    { id: 'deepseek-v4-flash', object: 'model', owned_by: 'deepseek' },
                    { id: 'deepseek-v4-pro', object: 'model', owned_by: 'deepseek' },
                ]
            };
        }
    }

    async getUsageLimits() {
        try {
            const response = await this.axiosInstance.get('/user/balance');
            return response.data;
        } catch (error) {
            logger.error('[DeepSeek] Failed to get user balance:', error.message);
            throwReadableDeepSeekError(error);
        }
    }
}
